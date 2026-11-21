from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import DateTime
from sqlalchemy import Text

from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    fullname = Column(String(100))
    email    = Column(String(150), unique=True, index=True)
    password = Column(String(255))

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    predictions = relationship(
        "Prediction",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), index=True)

    label              = Column(String(50))
    confidence         = Column(Float)
    probability_normal = Column(Float)
    probability_abnormal = Column(Float)

    # Patient snapshot (stored as JSON string for portability)
    patient_data = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    user = relationship("User", back_populates="predictions")
