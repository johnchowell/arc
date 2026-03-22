"""add domain, tld_group, content_category filter columns to pages

Revision ID: a1b2c3d4e5f6
Revises: c9c360d94d43
Create Date: 2026-03-03 22:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'c9c360d94d43'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("pages", sa.Column("domain", sa.String(253), nullable=True))
    op.add_column("pages", sa.Column("tld_group", sa.String(10), nullable=True))
    op.add_column("pages", sa.Column("content_category", sa.String(20), nullable=True))
    op.create_index("ix_pages_domain", "pages", ["domain"])
    op.create_index("ix_pages_tld_group", "pages", ["tld_group"])
    op.create_index("ix_pages_content_category", "pages", ["content_category"])


def downgrade() -> None:
    op.drop_index("ix_pages_content_category", "pages")
    op.drop_index("ix_pages_tld_group", "pages")
    op.drop_index("ix_pages_domain", "pages")
    op.drop_column("pages", "content_category")
    op.drop_column("pages", "tld_group")
    op.drop_column("pages", "domain")
