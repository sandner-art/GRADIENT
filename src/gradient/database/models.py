# src/gradient/database/models.py (NEW FILE)

import datetime
import json
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey, JSON, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Use SQLAlchemy's Declarative Base
Base = declarative_base()

class ExperimentRun(Base):
    """Represents a single execution run (e.g., a T2I call, a grid execution)."""
    __tablename__ = 'experiment_runs'

    id = Column(Integer, primary_key=True) # Auto-incrementing ID
    run_uuid = Column(String, unique=True, index=True, nullable=False) # Unique identifier for the run
    run_type = Column(String, nullable=False, index=True) # e.g., "text_to_image", "xyz_grid", "llm_chat", "latent_viz"
    start_time = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default="running", nullable=False, index=True) # running, completed, failed
    base_parameters = Column(JSON, nullable=True) # Store the initial request parameters (e.g., Text2ImageRequest dict)
    user_notes = Column(Text, nullable=True) # Optional user-added notes

    # Relationship to generated outputs
    outputs = relationship("GeneratedOutput", back_populates="run")

    def __repr__(self):
        return f"<ExperimentRun(id={self.id}, uuid='{self.run_uuid}', type='{self.run_type}', status='{self.status}')>"


class GeneratedOutput(Base):
    """Represents a single output file generated during a run."""
    __tablename__ = 'generated_outputs'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('experiment_runs.id'), nullable=False, index=True) # Link to the run
    output_type = Column(String, nullable=False, index=True) # e.g., "image", "latent_map", "plot", "log"
    file_path = Column(String, nullable=False, unique=True) # Relative or absolute path to the output file
    parameters = Column(JSON, nullable=True) # Specific parameters used for *this* output (e.g., seed, grid coords)
    output_metadata = Column(JSON, nullable=True, name="metadata") # Use 'name' for DB column if needed, Any extra metadata (e.g., performance, safety check status)
    creation_time = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    user_rating = Column(Integer, nullable=True) # Optional rating (e.g., 1-5)
    user_tags = Column(String, nullable=True) # Optional comma-separated tags

    # Relationship back to the run
    run = relationship("ExperimentRun", back_populates="outputs")

    def __repr__(self):
        return f"<GeneratedOutput(id={self.id}, run_id={self.run_id}, type='{self.output_type}', path='{self.file_path}')>"

# --- Add other models later ---
# class ParameterPreset(Base): ...
# class Workflow(Base): ...
# class ModelRegistry(Base): ...