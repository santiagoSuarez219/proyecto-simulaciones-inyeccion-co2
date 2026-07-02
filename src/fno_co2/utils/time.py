from datetime import datetime, timedelta


def get_next_pause_datetime(pause_hour: int) -> datetime:
    now = datetime.now()
    pause_dt = now.replace(hour=pause_hour, minute=0, second=0, microsecond=0)
    if now >= pause_dt:
        pause_dt = pause_dt + timedelta(days=1)
    return pause_dt
