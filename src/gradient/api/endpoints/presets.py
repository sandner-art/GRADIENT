# src/gradient/api/endpoints/presets.py

import datetime  # Ensure datetime is imported
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import Path as FastApiPath
from pydantic import BaseModel, ConfigDict, Field  # Import ConfigDict for Pydantic v2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from gradient.database.models import ModelPreset

# Import the context manager function directly
from gradient.database.session import db_session_dependency

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models for API ---


class PresetSetting(BaseModel):
    """Subset of parameters typically saved in a preset."""

    negative_prompt: str | None = None
    height: int | None = None
    width: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None  # Keep as float for now
    scheduler_id: str | None = None
    strength: float | None = None  # For Img2Img presets
    # vae_id: Optional[str] = None

    # Allow extra fields not explicitly defined above using model_config in Pydantic v2
    model_config = ConfigDict(extra="allow")


class PresetBase(BaseModel):
    model_identifier: str = Field(..., description="Model ID (path, Hub ID) this preset applies to")
    preset_name: str = Field(..., min_length=1, description="User-friendly name for the preset")
    settings: PresetSetting = Field(
        ..., description="Dictionary of parameter settings for this preset"
    )
    description: str | None = None


class PresetCreate(PresetBase):
    pass  # No additional fields needed for creation


class PresetRead(PresetBase):
    id: int
    is_user_defined: bool
    creation_time: datetime.datetime
    last_used_time: datetime.datetime | None = None

    # Use model_config for Pydantic v2 ORM mode
    model_config = ConfigDict(from_attributes=True)


# --- API Endpoints (Explicit async with workaround) ---


@router.post(
    "/presets",
    response_model=PresetRead,
    status_code=201,
    summary="Create a new user-defined preset",
    tags=["Presets"],
)
async def create_preset(
    preset_data: PresetCreate,
    db: AsyncSession = Depends(db_session_dependency),
):
    """Saves a new preset associated with a model identifier."""
    logger.info(
        f"Creating preset '{preset_data.preset_name}' for model '{preset_data.model_identifier}'"
    )
    try:
        # Check for duplicates using the session object 'db'
        stmt = select(ModelPreset).where(
            ModelPreset.model_identifier == preset_data.model_identifier,
            ModelPreset.preset_name == preset_data.preset_name,
        )
        result = await db.execute(stmt)
        existing = result.scalars().first()

        if existing:
            logger.warning(
                f"Preset '{preset_data.preset_name}' already exists for model '{preset_data.model_identifier}'."
            )
            # Raise HTTPException - rollback will be handled by context manager on exception
            raise HTTPException(
                status_code=409,
                detail=f"Preset '{preset_data.preset_name}' already exists for model '{preset_data.model_identifier}'.",
            )

        # Create new preset object
        db_preset = ModelPreset(
            model_identifier=preset_data.model_identifier,
            preset_name=preset_data.preset_name,
            settings=preset_data.settings.model_dump(exclude_unset=True),  # Store settings dict
            description=preset_data.description,
            is_user_defined=True,  # API creates user presets
        )
        db.add(db_preset)
        await db.commit()  # commit before responding so the preset is immediately listable
        await db.refresh(db_preset)
        logger.info(f"Preset added to session, ID (after flush): {db_preset.id}")
        # Commit will happen automatically by context manager if no exception during 'yield'
        return db_preset
    except HTTPException:
        # Re-raise HTTP exceptions so FastAPI handles them correctly
        raise
    except Exception as e:
        logger.error(f"Error during preset creation DB operation: {e}", exc_info=True)
        # Rollback is handled by context manager on any exception during 'yield'
        # Raise a generic 500 for other DB errors
        raise HTTPException(
            status_code=500, detail=f"Internal server error saving preset: {e}"
        ) from e


@router.get(
    "/presets",
    response_model=list[PresetRead],
    summary="List presets, optionally filtered by model ID",
    tags=["Presets"],
)
async def list_presets(
    model_identifier: str | None = Query(None, description="Filter presets by model identifier"),
    db: AsyncSession = Depends(db_session_dependency),
):
    """Retrieves a list of presets, optionally filtering by model."""
    logger.debug(f"Listing presets, filter model_identifier='{model_identifier}'")
    try:
        stmt = select(ModelPreset)
        if model_identifier:
            stmt = stmt.where(ModelPreset.model_identifier == model_identifier)
        stmt = stmt.order_by(ModelPreset.is_user_defined.asc(), ModelPreset.preset_name.asc())

        result = await db.execute(stmt)
        presets = result.scalars().all()
        return presets
    except Exception as e:
        logger.error(f"Error listing presets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error listing presets.") from e


@router.get(
    "/presets/{preset_id}",
    response_model=PresetRead,
    summary="Get a specific preset by its ID",
    tags=["Presets"],
)
async def get_preset(
    preset_id: int = FastApiPath(..., gt=0, description="ID of the preset to retrieve"),
    db: AsyncSession = Depends(db_session_dependency),
):
    """Retrieves details of a single preset by its database ID."""
    logger.debug(f"Fetching preset with ID: {preset_id}")
    try:
        # Use db.get for primary key lookup - efficient
        db_preset = await db.get(ModelPreset, preset_id)

        if not db_preset:
            logger.warning(f"Preset with ID {preset_id} not found.")
            raise HTTPException(status_code=404, detail=f"Preset with ID {preset_id} not found.")
        return db_preset
    except HTTPException:
        raise  # Re-raise
    except Exception as e:
        logger.error(f"Error fetching preset ID {preset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error fetching preset.") from e


@router.delete(
    "/presets/{preset_id}",
    status_code=204,  # No Content
    summary="Delete a user-defined preset by ID",
    tags=["Presets"],
)
async def delete_preset(
    preset_id: int = FastApiPath(..., gt=0, description="ID of the preset to delete"),
    db: AsyncSession = Depends(db_session_dependency),
):
    """Deletes a specific user-defined preset."""
    logger.info(f"Attempting to delete preset with ID: {preset_id}")
    try:
        db_preset = await db.get(ModelPreset, preset_id)

        if not db_preset:
            logger.warning(f"Preset ID {preset_id} not found for deletion.")
            raise HTTPException(status_code=404, detail=f"Preset with ID {preset_id} not found.")

        # Optional: Add user check here if multi-user
        # if db_preset.user_id != current_user.id: raise HTTPException(status_code=403, ...)

        if not db_preset.is_user_defined:
            logger.warning(f"Attempted delete non-user preset ID: {preset_id}")
            raise HTTPException(status_code=403, detail="Cannot delete system-defined presets.")

        await db.delete(db_preset)
        await db.commit()
        logger.info(f"Successfully marked preset ID {preset_id} for deletion.")
        # Commit handled by context manager on successful exit
        return None  # Return None for 204 response
    except HTTPException:
        raise  # Re-raise
    except Exception as e:
        logger.error(f"Error deleting preset ID {preset_id}: {e}", exc_info=True)
        # Rollback handled by context manager
        raise HTTPException(status_code=500, detail="Internal server error deleting preset.") from e


# TODO: Add PUT /presets/{preset_id} for updating user presets later if needed
