"""
tasks/benchmark_tasks.py

Five complex tasks designed to stress-test long-horizon memory within a single
reasoning chain. Each task requires the model to:
  - Establish facts in early steps
  - Reference and build on those facts in later steps
  - Maintain constraints set at step 1 through to step 5
  - Catch contradictions between steps (self-verification signal)

A model with weak within-session memory will contradict its own earlier
reasoning by step 4 or 5. That contradiction rate is the core metric.
"""

TASKS = [
    {
        "id": "T1",
        "title": "Cascading schema migration across three dependent services",
        "difficulty": "hard",
        "category": "distributed_systems",
        "memory_trap": "constraint established in step 1 must be respected by proposed fix in step 5",
        "description": """
You are the lead engineer reviewing a planned database schema migration.
The system has three services: UserService, OrderService, and AnalyticsService.

Current schema (PostgreSQL):
  users:  id (PK), email, created_at
  orders: id (PK), user_id (FK → users.id), total, status, created_at
  events: id (PK), user_id (FK → users.id), event_type, payload JSONB, created_at

Migration goal: add a tenant_id column to all three tables to support multi-tenancy.

Hard constraint established now — remember this for every subsequent step:
  The migration must be zero-downtime. The system processes 50,000 orders/hour
  and cannot tolerate any table locks lasting more than 200ms.

Step 1:
  Identify all foreign key relationships that will need updating.
  State explicitly how many FK constraints exist across the three tables.
  
Step 2:
  Write the migration SQL for adding tenant_id to the users table only.
  Justify every DDL choice against the 200ms lock constraint you noted in step 1.
  
Step 3:
  The orders table has 800 million rows. Write the migration for orders.tenant_id.
  Your approach must differ from step 2. Explain why, referencing the row count.
  
Step 4:
  AnalyticsService reads from events with a query that runs every 30 seconds:
    SELECT user_id, COUNT(*) FROM events WHERE created_at > NOW() - INTERVAL '1 hour'
    GROUP BY user_id;
  Will adding tenant_id break this query's performance? Prove your answer using
  the existing index structure implied by the schema.
  
Step 5:
  A junior engineer proposes: "Just add NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
  to all three tables in a single transaction."
  Evaluate this against every constraint and finding from steps 1-4.
  Be specific about which of your earlier findings it violates.
""",
        "expected_steps": 5,
        "constraint_to_track": "200ms lock limit — must appear in steps 2, 3, and 5 evaluation",
        "memory_failure_signal": "Step 5 approves the NOT NULL DEFAULT proposal without citing the lock constraint",
        "verification_hints": [
            "2 FK constraints: orders.user_id and events.user_id",
            "ADD COLUMN with DEFAULT is instant in PG 11+ but NOT NULL DEFAULT on large table causes rewrite",
            "800M rows needs background migration: add nullable, backfill in batches, then add constraint",
            "existing index on created_at doesn't include tenant_id — query unaffected",
            "single transaction NOT NULL DEFAULT violates 200ms constraint on orders (800M rows)",
        ],
    },

    {
        "id": "T2",
        "title": "API rate limiter with sliding window — find all the bugs",
        "difficulty": "hard",
        "category": "concurrency",
        "memory_trap": "three bugs found across steps 1-3 must all be cited in the final fix",
        "description": """
Review this Redis-backed sliding window rate limiter used in a production API
serving 2M requests/day. Your job is a full adversarial review.

```python
import time
import redis

r = redis.Redis()

def is_allowed(user_id: str, limit: int = 100, window: int = 3600) -> bool:
    key = f"ratelimit:{user_id}"
    now = time.time()
    window_start = now - window

    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)
    pipe.zadd(key, {str(now): now})
    pipe.zcard(key)
    pipe.expire(key, window)
    results = pipe.execute()

    count = results[2]
    return count <= limit
```

Step 1:
  Identify the race condition. Show the exact sequence of two concurrent
  requests from the same user that allows both to succeed when only one should.
  Name the Redis commands involved and the window of vulnerability.

Step 2:
  The zadd key uses str(now) as the member and now as the score.
  What happens when two requests arrive within the same millisecond?
  How many requests does this silently drop? Prove it.

Step 3:
  The expire is set to `window` seconds from now on every request.
  Describe the scenario where this causes a user's rate limit window
  to never reset. Give a concrete request timing that triggers it.

Step 4:
  Fix the race condition from step 1 only, using a Lua script.
  Your fix must not introduce the bug from step 2 or step 3.
  Show the complete Lua script.

Step 5:
  Now write the complete fixed implementation that resolves all three bugs
  from steps 1, 2, and 3. Reference each bug by its step number as you fix it.
  Do not introduce any new issues.
""",
        "expected_steps": 5,
        "constraint_to_track": "three distinct bugs from steps 1, 2, 3 — all three must be cited in step 5",
        "memory_failure_signal": "Step 5 fix addresses only 1 or 2 of the 3 bugs without acknowledging the omission",
        "verification_hints": [
            "pipeline is not atomic — two concurrent pipelines can both read count=99 before either increments",
            "same-millisecond requests: str(now) collision causes zadd to overwrite, silently dropping one request",
            "expire resets on every request — active user never has the key expire, window never fully clears",
            "Lua script must wrap zremrangebyscore + zadd + zcard atomically",
            "fix step 2: use unique member (uuid or user_id:timestamp:random), keep score=now",
        ],
    },

    {
        "id": "T3",
        "title": "Async task queue deadlock under backpressure",
        "difficulty": "expert",
        "category": "async_concurrency",
        "memory_trap": "root cause identified in step 2 must be the basis for the fix in step 4",
        "description": """
A FastAPI service uses an internal async task queue for background processing.
Under load (>500 req/s), the service deadlocks completely — all requests hang
indefinitely. The deadlock is 100% reproducible above the threshold.

```python
import asyncio
from fastapi import FastAPI

app = FastAPI()

queue = asyncio.Queue(maxsize=100)
results = {}

async def worker():
    while True:
        task_id, coro = await queue.get()
        result = await coro
        results[task_id] = result
        queue.task_done()

@app.on_event("startup")
async def startup():
    asyncio.create_task(worker())   # single worker

@app.post("/process")
async def process(data: dict):
    task_id = data["id"]
    
    async def compute():
        # Simulate work that may enqueue follow-up tasks
        if data.get("chain"):
            await queue.put((task_id + "_followup", some_followup_coro()))
        return {"processed": True}
    
    await queue.put((task_id, compute()))
    await queue.join()              # wait for ALL tasks to complete
    return results.get(task_id)
```

Step 1:
  Trace the exact execution path for a chained request (data["chain"]=True)
  that causes the deadlock. Name the asyncio primitives involved.

Step 2:
  Why does this only deadlock above 500 req/s and not at lower traffic?
  Your answer must reference the queue's maxsize=100 and the single worker.
  State the precise condition that must be true simultaneously.

Step 3:
  A colleague proposes: increase maxsize to 10000.
  Does this fix the deadlock? Justify your answer using your findings from step 2.
  If it does not fix it, explain what it changes and what it does not.

Step 4:
  Fix the deadlock. Your fix must not change the external API contract
  (same endpoint, same response format). Reference your root cause from step 2.

Step 5:
  The fix you proposed in step 4 — does it handle the case where
  some_followup_coro() itself enqueues further tasks (depth > 2)?
  If not, extend the fix. If yes, prove it handles arbitrary chain depth.
""",
        "expected_steps": 5,
        "constraint_to_track": "root cause from step 2 must be explicitly cited in step 4 fix",
        "memory_failure_signal": "Step 4 fix doesn't reference the queue.join() + single worker + maxsize interaction",
        "verification_hints": [
            "compute() enqueues follow-up, but worker is occupied running compute() — follow-up can't be dequeued",
            "queue.join() blocks the request handler, which blocks the event loop, which blocks the worker",
            "below 500 req/s: queue rarely full so follow-up fits; above: queue full, put() blocks, deadlock",
            "maxsize=10000 delays but doesn't fix — same deadlock once queue fills",
            "fix: remove queue.join(), use asyncio.Event or Future per task_id for result signalling",
        ],
    },

    {
        "id": "T4",
        "title": "SQLAlchemy ORM query producing wrong results silently",
        "difficulty": "hard",
        "category": "data_integrity",
        "memory_trap": "two distinct silent failure modes found in steps 1 and 2 must both appear in step 5",
        "description": """
A billing system uses SQLAlchemy ORM. Finance has flagged that monthly invoice
totals are occasionally wrong by small amounts — sometimes over, sometimes under.
The bug has been in production for 6 months. No exceptions are raised.

```python
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import Invoice, LineItem, Adjustment

def get_monthly_total(db: Session, customer_id: int, year: int, month: int) -> float:
    # Get all invoices for the month
    invoices = db.query(Invoice).filter(
        Invoice.customer_id == customer_id,
        func.extract('year', Invoice.created_at) == year,
        func.extract('month', Invoice.created_at) == month,
    ).all()

    total = 0.0
    for invoice in invoices:
        # Sum line items
        line_total = db.query(func.sum(LineItem.amount)).filter(
            LineItem.invoice_id == invoice.id
        ).scalar()
        total += line_total or 0

        # Apply adjustments (credits/debits)
        adjustments = db.query(Adjustment).filter(
            Adjustment.invoice_id == invoice.id,
            Adjustment.applied == True
        ).all()
        for adj in adjustments:
            total += adj.amount   # positive = debit, negative = credit

    return total

def apply_adjustment(db: Session, invoice_id: int, amount: float, reason: str):
    adj = Adjustment(invoice_id=invoice_id, amount=amount, reason=reason,
                     applied=False)
    db.add(adj)
    db.flush()
    adj.applied = True
    db.commit()
    return adj
```

Step 1:
  Identify the floating point issue. For a customer with 47 line items averaging
  $23.99, calculate the maximum possible accumulated error. Show your working.

Step 2:
  The apply_adjustment function has a race condition that causes adjustments
  to be double-counted in get_monthly_total. Describe the exact sequence
  of two concurrent transactions that triggers it.

Step 3:
  There is a third bug unrelated to steps 1 and 2. It involves the SQLAlchemy
  identity map and the db.flush() call. Identify it and explain what data
  state it can produce.

Step 4:
  Fix step 1 only: rewrite get_monthly_total to eliminate floating point error.
  Your fix must produce results accurate to the cent on any realistic input.

Step 5:
  Write the complete fixed implementation for both functions.
  Explicitly label each fix as: [Step 1 fix], [Step 2 fix], [Step 3 fix].
  Do not merge fixes silently — label them so a code reviewer can verify each.
""",
        "expected_steps": 5,
        "constraint_to_track": "three bugs from steps 1, 2, 3 must all be labelled in step 5",
        "memory_failure_signal": "Step 5 merges fixes without labelling or misses one of the three bugs",
        "verification_hints": [
            "float accumulation: 47 * 23.99 = $1127.53, float error up to ~$0.02 on this size",
            "use Decimal or store amounts as integer cents",
            "race: transaction A reads applied=False, transaction B commits applied=True, A overwrites with applied=True creating duplicate",
            "fix: SELECT FOR UPDATE or optimistic locking on Adjustment",
            "flush() before commit: identity map returns the unflushed object to concurrent queries in same session",
        ],
    },

    {
        "id": "T5",
        "title": "Distributed tracing gap causing silent data loss in event pipeline",
        "difficulty": "expert",
        "category": "observability",
        "memory_trap": "failure scenario constructed in step 1 must be used as test case in step 4 and step 5",
        "description": """
An event processing pipeline ingests 10M events/day. The team noticed that
approximately 0.003% of events vanish without a trace — no error logs,
no dead letter queue entries, no metrics. The pipeline is:

  Producer → Kafka topic → Consumer (Python) → Processor → PostgreSQL

```python
import asyncio
import json
from aiokafka import AIOKafkaConsumer
from sqlalchemy.ext.asyncio import AsyncSession

consumer = AIOKafkaConsumer(
    'events',
    bootstrap_servers='kafka:9092',
    group_id='processor',
    enable_auto_commit=True,       # commits offset automatically
    auto_commit_interval_ms=5000,
)

async def process_event(session: AsyncSession, event: dict) -> bool:
    try:
        result = await session.execute(
            insert(Event).values(**event).on_conflict_do_nothing()
        )
        await session.commit()
        return result.rowcount > 0
    except Exception as e:
        await session.rollback()
        return False

async def consume():
    await consumer.start()
    async for msg in consumer:
        event = json.loads(msg.value)
        async with AsyncSession(engine) as session:
            success = await process_event(session, event)
            if not success:
                print(f"Failed to process event {event.get('id')}")
```

Step 1:
  Construct a precise failure scenario where an event is consumed, processed
  successfully (rowcount=1), but the offset is never committed to Kafka.
  Name the exact conditions required. This scenario will be your test case
  for the remaining steps.

Step 2:
  The on_conflict_do_nothing() silently swallows a class of failures.
  Describe a scenario where an event appears to succeed (rowcount=0 is not
  checked as failure) but the data is wrong. How does this interact with
  the auto-commit behaviour from step 1?

Step 3:
  enable_auto_commit=True commits offsets on a timer, not on successful
  processing. Calculate the maximum number of events that can be silently
  lost in the scenario from step 1 if the consumer crashes at the worst
  possible moment. Show the maths using auto_commit_interval_ms=5000
  and an ingestion rate of 10M events/day.

Step 4:
  Fix the offset commit issue. Your fix must handle the failure scenario
  you constructed in step 1 — reference it explicitly.
  Do not change the external Kafka topic or producer.

Step 5:
  The fix you wrote in step 4 introduces a new problem under high load:
  processing latency spikes from 5ms to 800ms per event. Diagnose why
  and fix it while preserving the correctness guarantee from step 4.
  Reference step 1 again to confirm your optimised fix still handles it.
""",
        "expected_steps": 5,
        "constraint_to_track": "step 1 failure scenario must be explicitly referenced in steps 4 and 5",
        "memory_failure_signal": "Steps 4 and 5 propose fixes without citing the step 1 scenario as validation",
        "verification_hints": [
            "crash between session.commit() and auto-commit timer firing — offset never written",
            "on_conflict_do_nothing rowcount=0 misread as 'already processed' but could be schema mismatch",
            "10M/day = ~115 events/sec, 5s window = ~577 events at risk of reprocessing, not loss",
            "fix: disable auto_commit, manually commit offset after successful db commit",
            "latency spike: manual commit per message = synchronous Kafka roundtrip per event; fix with batch commit",
        ],
    },
]
