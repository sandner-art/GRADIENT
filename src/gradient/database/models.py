# src/gradient/database/models.py

import datetime
import json # Keep json import if needed, though SQLAlchemy handles JSON type
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey, JSON, Boolean, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from typing import Optional # Ensure Optional is imported if needed in type hints, not strictly required for models here

# Use SQLAlchemy's Declarative Base
Base = declarative_base()

class ExperimentRun(Base):
    """Represents a single execution run (e.g., a T2I call, a grid execution)."""
    __tablename__ = 'experiment_runs'

    id = Column(Integer, primary_key=True) # Auto-incrementing ID
    run_uuid = Column(String, unique=True, index=True, nullable=False) # Unique identifier for the run
    run_type = Column(String, nullable=False, index=True) # e.g., "text_to_image", "image_to_image", "xyz_grid", "llm_chat", "latent_viz"
    start_time = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default="running", nullable=False, index=True) # running, completed, failed
    # Store the initial request parameters (e.g., Text2ImageRequest dict)
    # For grid, this might store base + axes definitions
    base_parameters = Column(JSON, nullable=True)
    user_notes = Column(Text, nullable=True) # Optional user-added notes

    # Relationship to generated outputs for this run
    outputs = relationship("GeneratedOutput", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ExperimentRun(id={self.id}, uuid='{self.run_uuid}', type='{self.run_type}', status='{self.status}')>"


class GeneratedOutput(Base):
    """Represents a single output file generated during a run."""
    __tablename__ = 'generated_outputs'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('experiment_runs.id'), nullable=False, index=True) # Link to the run
    output_type = Column(String, nullable=False, index=True) # e.g., "image", "latent_map", "plot", "log"
    file_path = Column(String, nullable=False, unique=True) # Relative or absolute path to the output file
    # Specific parameters used for *this* output (e.g., seed, grid coords, specific LoRA weight)
    parameters = Column(JSON, nullable=True)
    # Use 'name' for DB column if needed, Any extra metadata (e.g., performance, safety check status)
    output_metadata = Column(JSON, nullable=True, name="metadata")
    creation_time = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    user_rating = Column(Integer, nullable=True) # Optional rating (e.g., 1-5)
    user_tags = Column(String, nullable=True) # Optional comma-separated tags

    # Relationship back to the run
    run = relationship("ExperimentRun", back_populates="outputs")

    def __repr__(self):
        return f"<GeneratedOutput(id={self.id}, run_id={self.run_id}, type='{self.output_type}', path='{self.file_path}')>"


class ModelPreset(Base):
    """Stores named parameter presets associated with specific model identifiers."""
    __tablename__ = 'model_presets'

    id = Column(Integer, primary_key=True)
    # Identifier for the model this preset is for (can be path, HF ID, etc.)
    # Needs to match the 'model_id' used in generation requests
    model_identifier = Column(String, index=True, nullable=False)
    preset_name = Column(String, nullable=False) # e.g., "Default Fast", "User Anime v1"
    # Store the actual settings dictionary (subset of Text2ImageRequest/etc.)
    settings = Column(JSON, nullable=False)
    # Differentiate between built-in/recommended and user presets
    is_user_defined = Column(Boolean, default=True, nullable=False)
    # Optional description
    description = Column(Text, nullable=True)
    # Optional: Link to user if implementing multi-user later
    # user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    creation_time = Column(DateTime, default=datetime.datetime.utcnow)
    last_used_time = Column(DateTime, nullable=True) # Maybe update this when preset is used?

    # Ensure a user cannot have two presets with the same name for the same model
    # If adding users, include user_id in the constraint
    __table_args__ = (UniqueConstraint('model_identifier', 'preset_name', name='_model_preset_uc'),)

    def __repr__(self):
        user_tag = "[User]" if self.is_user_defined else "[System]"
        return f"<ModelPreset(id={self.id}, model='{self.model_identifier}', name='{self.preset_name}' {user_tag})>"

# --- Add other models later ---
# class ParameterPreset(Base): ... # ModelPreset replaces this idea generally
# class Workflow(Base): ...
# class ModelRegistry(Base): ... # Could track known models, metadata, etc.