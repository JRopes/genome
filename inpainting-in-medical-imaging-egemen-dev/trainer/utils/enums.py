from enum import Enum


class Stage(str, Enum):
    INTERMISSION = "intermission"
    TRAINING = "train"
    VALIDATION = "val"
    LOADING_CP = "loading"
    SAVING_CP = "saving"
