from enum import Enum

class LessonType(Enum):
    Practice = "practice",
    Lecture = "lecture"

class Lesson:
    def __init__(self, lesson_type, teacher, groups, subject, lesson_num):
        self.type = lesson_type
        self.teacher = teacher
        self.groups = groups
        self.subject = subject
        self.lessonNum = lesson_num