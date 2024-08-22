from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    ForeignKey,
    Table,
    DateTime,
    Boolean,
    Float,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, backref
import datetime

Base = declarative_base()

device_user_association = Table(
    "device_user",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("device_id", Integer, ForeignKey("devices.id")),
)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    devices = relationship(
        "Device", secondary=device_user_association, back_populates="users"
    )
    security_logs = relationship("SecurityLog", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"


class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_name = Column(String(100), nullable=False)
    device_type = Column(String(50), nullable=False)
    ip_address = Column(String(45), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    users = relationship(
        "User", secondary=device_user_association, back_populates="devices"
    )
    events = relationship("SecurityEvent", back_populates="device")

    def __repr__(self):
        return f"<Device(id={self.id}, device_name={self.device_name}, device_type={self.device_type})>"


class SecurityEvent(Base):
    __tablename__ = "security_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)
    severity = Column(Integer, nullable=False)
    description = Column(String(500))
    occurred_at = Column(DateTime, default=datetime.datetime.utcnow)
    device_id = Column(Integer, ForeignKey("devices.id"))

    device = relationship("Device", back_populates="events")
    logs = relationship("SecurityLog", back_populates="event")

    def __repr__(self):
        return f"<SecurityEvent(id={self.id}, event_type={self.event_type}, severity={self.severity})>"


class SecurityLog(Base):
    __tablename__ = "security_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    success = Column(Boolean, default=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    event_id = Column(Integer, ForeignKey("security_events.id"))

    user = relationship("User", back_populates="security_logs")
    event = relationship("SecurityEvent", back_populates="logs")

    def __repr__(self):
        return (
            f"<SecurityLog(id={self.id}, action={self.action}, success={self.success})>"
        )


class DataPacket(Base):
    __tablename__ = "data_packets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_ip = Column(String(45), nullable=False)
    destination_ip = Column(String(45), nullable=False)
    packet_size = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<DataPacket(id={self.id}, source_ip={self.source_ip}, destination_ip={self.destination_ip})>"


class AnomalyDetection(Base):
    __tablename__ = "anomaly_detection"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    parameters = Column(String(1000), nullable=False)
    accuracy = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<AnomalyDetection(id={self.id}, model_name={self.model_name}, accuracy={self.accuracy})>"


# Database configuration
DATABASE_URL = "sqlite:///smart_sound_guardian.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

Base.metadata.create_all(engine)
