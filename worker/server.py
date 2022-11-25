from typing import List
import multiprocessing as mp

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI()


class JobConfig(BaseModel):
    type: str = Field(const=True, default="test")


class InterpolationAnimationJobConfig(BaseModel):
    type: str = Field(const=True, default="interpolation_animation")


class Job(BaseModel):
    config: JobConfig
    job_id: int = None

_jobs = []
_prefix = "/api/v1"
_current_id = 0


@app.post(_prefix + "/job", response_model=Job)
def add_job(job: Job) -> Job:
    global _current_id
    job.job_id = _current_id
    # TODO(j.swannack): concurrency issues, this is ok 
    # because it's proabably just me using it
    _current_id += 1

    _jobs.append(job)

    return job 

@app.get(_prefix + "/job", response_model=List[Job])
def get_jobs() -> List[Job]:
    return _jobs


if __name__ == "__main__":
    from uvicorn import run
    run(app, host="0.0.0.0", port=8888)
