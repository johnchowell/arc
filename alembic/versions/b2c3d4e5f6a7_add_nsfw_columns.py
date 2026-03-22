"""add nsfw_flag to pages and nsfw_score to image_embeddings

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-04 10:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("pages", sa.Column("nsfw_flag", sa.Boolean(), nullable=True))
    op.create_index("ix_pages_nsfw_flag", "pages", ["nsfw_flag"])
    op.add_column("image_embeddings", sa.Column("nsfw_score", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("image_embeddings", "nsfw_score")
    op.drop_index("ix_pages_nsfw_flag", "pages")
    op.drop_column("pages", "nsfw_flag")
