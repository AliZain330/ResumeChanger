from pydantic import BaseModel, Field, EmailStr
from typing import List


class Basics(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: EmailStr


class ExperienceItem(BaseModel):
    role: str = Field(min_length=1, max_length=80)
    company: str = Field(min_length=1, max_length=80)
    start: str = Field(min_length=1, max_length=20)
    end: str = Field(min_length=1, max_length=20)
    bullets: List[str] = Field(min_length=1, max_length=5)


class Resume(BaseModel):
    basics: Basics
    summary: str = Field(min_length=30, max_length=600)
    skills: List[str] = Field(default_factory=list, max_length=30)
    experience: List[ExperienceItem] = Field(default_factory=list, max_length=5)
