"""add links_harvested_at to pages

Revision ID: c9c360d94d43
Revises: 63bbc337c17a
Create Date: 2026-03-03 16:47:58.947674
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = 'c9c360d94d43'
down_revision: Union[str, None] = '63bbc337c17a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("pages", sa.Column("links_harvested_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("pages", "links_harvested_at")
