"""迁移 clips.clip_status：旧状态 → 新三状态体系"""
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import SessionLocal
from sqlalchemy import text

async def migrate():
    async with SessionLocal() as s:
        r1 = await s.execute(text(
            "UPDATE clips SET clip_status='matched', is_active=TRUE "
            "WHERE clip_status IN ('done','confirmed')"
        ))
        print(f"done/confirmed -> matched: {r1.rowcount} rows")

        r2 = await s.execute(text(
            "UPDATE clips SET clip_status='unmatched', is_active=FALSE "
            "WHERE clip_status IN ('skipped','pending','rejected')"
        ))
        print(f"skipped/pending/rejected -> unmatched: {r2.rowcount} rows")

        r3 = await s.execute(text(
            "UPDATE clips SET is_active=FALSE WHERE clip_status='failed'"
        ))
        print(f"failed is_active fix: {r3.rowcount} rows")

        r4 = await s.execute(text(
            "UPDATE clips SET is_active=FALSE WHERE clip_status='unmatched'"
        ))
        print(f"unmatched is_active fix: {r4.rowcount} rows")

        await s.commit()

        rows = (await s.execute(text(
            "SELECT clip_status, COUNT(*) as cnt FROM clips GROUP BY clip_status"
        ))).all()
        print("\n--- Migration result ---")
        for r in rows:
            print(f"  {r[0]}: {r[1]}")

asyncio.run(migrate())
