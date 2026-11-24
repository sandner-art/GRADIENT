# src/gradient/database/models.py

import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


# Use SQLAlchemy's Declarative Base
Base = declarative_base()


class ExperimentRun(Base):
    """Represents a single execution run (e.g., a T2I call, a grid execution)."""

    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True)  # Auto-incrementing ID
    run_uuid = Column(
        String, unique=True, index=True, nullable=False
    )  # Unique identifier for the run
    run_type = Column(
        String, nullable=False, index=True
    )  # e.g., "text_to_image", "image_to_image", "xyz_grid", "llm_chat", "latent_viz"
    start_time = Column(DateTime, default=_utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    status = Column(
        String, default="running", nullable=False, index=True
    )  # running, completed, failed
    # Store the initial request parameters (e.g., Text2ImageRequest dict)
    # For grid, this might store base + axes definitions
    base_parameters = Column(JSON, nullable=True)
    user_notes = Column(Text, nullable=True)  # Optional user-added notes

    # Relationship to generated outputs for this run
    outputs = relationship("GeneratedOutput", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ExperimentRun(id={self.id}, uuid='{self.run_uuid}', type='{self.run_type}', status='{self.status}')>"


class GeneratedOutput(Base):
    """Represents a single output file generated during a run."""

    __tablename__ = "generated_outputs"

    id = Column(Integer, primary_key=True)
    run_id = Column(
        Integer, ForeignKey("experiment_runs.id"), nullable=False, index=True
    )  # Link to the run
    output_type = Column(
        String, nullable=False, index=True
    )  # e.g., "image", "latent_map", "plot", "log"
    file_path = Column(
        String, nullable=False, unique=True
    )  # Relative or absolute path to the output file
    # Specific parameters used for *this* output (e.g., seed, grid coords, specific LoRA weight)
    parameters = Column(JSON, nullable=True)
    # Use 'name' for DB column if needed, Any extra metadata (e.g., performance, safety check status)
    output_metadata = Column(JSON, nullable=True, name="metadata")
    creation_time = Column(DateTime, default=_utcnow, nullable=False)
    user_rating = Column(Integer, nullable=True)  # Optional rating (e.g., 1-5)
    user_tags = Column(String, nullable=True)  # legacy comma-separated tags (see OutputTag)
    # Gallery fields (v0.5); copied out of `parameters` so they can be filtered and sorted
    favorite = Column(Boolean, default=False, nullable=False, server_default="0")
    notes = Column(Text, nullable=True)
    model_id = Column(String, nullable=True, index=True)
    prompt = Column(Text, nullable=True)
    seed = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

    # Relationship back to the run
    run = relationship("ExperimentRun", back_populates="outputs")
    tags = relationship("OutputTag", back_populates="output", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<GeneratedOutput(id={self.id}, run_id={self.run_id}, type='{self.output_type}', path='{self.file_path}')>"


class OutputTag(Base):
    """A tag on an output, with where it came from (you, Kurator, ArtAgents, an AI tagger)."""

    __tablename__ = "output_tags"

    id = Column(Integer, primary_key=True)
    output_id = Column(
        Integer, ForeignKey("generated_outputs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tag = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False, default="manual")  # manual | kurator | artagents | ai
    confidence = Column(Float, nullable=True)
    added = Column(DateTime, default=_utcnow, nullable=False)

    output = relationship("GeneratedOutput", back_populates="tags")
    __table_args__ = (UniqueConstraint("output_id", "tag", "source", name="_output_tag_uc"),)


class OutputCaption(Base):
    """A caption for an output, one per source (e.g. an ArtAgents caption file, synced back)."""

    __tablename__ = "output_captions"

    id = Column(Integer, primary_key=True)
    output_id = Column(
        Integer, ForeignKey("generated_outputs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    updated = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)
    __table_args__ = (UniqueConstraint("output_id", "source", name="_output_caption_uc"),)


class Project(Base):
    """A folder of exported images shared with other tools (Kurator, ArtAgents, digiKam ...)."""

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    uuid = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    path = Column(String, unique=True, nullable=False)
    created = Column(DateTime, default=_utcnow, nullable=False)
    updated = Column(DateTime, default=_utcnow, nullable=False)  # last export
    last_sync = Column(DateTime, nullable=True)
    options = Column(JSON, nullable=True)  # export options used last time


class ProjectItem(Base):
    """An output exported into a project folder."""

    __tablename__ = "project_items"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    output_id = Column(
        Integer, ForeignKey("generated_outputs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_name = Column(String, nullable=False)
    __table_args__ = (UniqueConstraint("project_id", "output_id", name="_project_item_uc"),)


class QueueJob(Base):
    """A job of the render queue (kept for the queue panel's history and retries)."""

    __tablename__ = "queue_jobs"

    id = Column(String, primary_key=True)
    kind = Column(String, nullable=False, index=True)
    label = Column(String, nullable=False, default="")
    status = Column(String, nullable=False, index=True)
    priority = Column(Integer, nullable=False, default=0)
    payload = Column(JSON, nullable=True)  # without inline images
    created = Column(Float, nullable=False)
    started = Column(Float, nullable=True)
    finished = Column(Float, nullable=True)
    run_uuid = Column(String, nullable=True, index=True)
    error = Column(Text, nullable=True)
    result = Column(JSON, nullable=True)  # summary: run_uuid, output ids, timings
    meta = Column(JSON, nullable=True)  # group, tags, origin given to POST /queue


class ModelPreset(Base):
    """Stores named parameter presets associated with specific model identifiers."""

    __tablename__ = "model_presets"

    id = Column(Integer, primary_key=True)
    # Identifier for the model this preset is for (can be path, HF ID, etc.)
    # Needs to match the 'model_id' used in generation requests
    model_identifier = Column(String, index=True, nullable=False)
    preset_name = Column(String, nullable=False)  # e.g., "Default Fast", "User Anime v1"
    # Store the actual settings dictionary (subset of Text2ImageRequest/etc.)
    settings = Column(JSON, nullable=False)
    # Differentiate between built-in/recommended and user presets
    is_user_defined = Column(Boolean, default=True, nullable=False)
    # Optional description
    description = Column(Text, nullable=True)
    # Optional: Link to user if implementing multi-user later
    # user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    creation_time = Column(DateTime, default=_utcnow)
    last_used_time = Column(DateTime, nullable=True)  # Maybe update this when preset is used?

    # Ensure a user cannot have two presets with the same name for the same model
    # If adding users, include user_id in the constraint
    __table_args__ = (UniqueConstraint("model_identifier", "preset_name", name="_model_preset_uc"),)

    def __repr__(self):
        user_tag = "[User]" if self.is_user_defined else "[System]"
        return f"<ModelPreset(id={self.id}, model='{self.model_identifier}', name='{self.preset_name}' {user_tag})>"


# --- Add other models later ---
# class ParameterPreset(Base): ... # ModelPreset replaces this idea generally
# class Workflow(Base): ...
# class ModelRegistry(Base): ... # Could track known models, metadata, etc.
