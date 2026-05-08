"""
Kafka consumer for real-time complaint stream ingestion.
Feeds into the Celery task queue for async agent pipeline execution.
"""
from __future__ import annotations

import json
import logging
import signal
import sys
from typing import Callable, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings
from src.schemas.models import ComplaintSubmitRequest

logger = logging.getLogger(__name__)


class ComplaintKafkaConsumer:
    def __init__(self, on_message: Callable[[dict], None]) -> None:
        cfg = get_settings()
        self._on_message = on_message
        self._running = False

        self._consumer = Consumer({
            "bootstrap.servers": cfg.kafka_bootstrap_servers,
            "group.id": cfg.kafka_consumer_group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": 30000,
        })

        self._dlq_producer = Producer({
            "bootstrap.servers": cfg.kafka_bootstrap_servers,
            "acks": "all",
        })

        self._topic = cfg.kafka_complaints_topic
        self._dlq_topic = cfg.kafka_dlq_topic

        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def start(self) -> None:
        self._consumer.subscribe([self._topic])
        self._running = True
        logger.info("Kafka consumer started, subscribed to %s", self._topic)

        try:
            while self._running:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise KafkaException(msg.error())

                self._process_message(msg)
        finally:
            self._consumer.close()
            logger.info("Kafka consumer shut down")

    def _process_message(self, msg) -> None:
        try:
            payload = json.loads(msg.value().decode("utf-8"))
            ComplaintSubmitRequest(**payload)  # validate schema
            self._on_message(payload)
            self._consumer.commit(msg)
        except Exception as exc:
            logger.error("Failed to process message offset=%d: %s", msg.offset(), exc)
            self._send_to_dlq(msg, str(exc))
            self._consumer.commit(msg)

    def _send_to_dlq(self, original_msg, error: str) -> None:
        dlq_payload = {
            "original_topic": self._topic,
            "original_offset": original_msg.offset(),
            "error": error,
            "raw_value": original_msg.value().decode("utf-8", errors="replace"),
        }
        self._dlq_producer.produce(
            self._dlq_topic,
            json.dumps(dlq_payload).encode("utf-8"),
        )
        self._dlq_producer.flush()

    def _shutdown(self, signum, frame) -> None:
        logger.info("Shutdown signal received")
        self._running = False


class ComplaintKafkaProducer:
    def __init__(self) -> None:
        cfg = get_settings()
        self._producer = Producer({
            "bootstrap.servers": cfg.kafka_bootstrap_servers,
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 1000,
        })
        self._topic = cfg.kafka_complaints_topic

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def send(self, payload: dict) -> None:
        self._producer.produce(
            self._topic,
            json.dumps(payload).encode("utf-8"),
            callback=self._delivery_report,
        )
        self._producer.flush()

    @staticmethod
    def _delivery_report(err, msg) -> None:
        if err:
            logger.error("Message delivery failed: %s", err)
        else:
            logger.debug("Message delivered to %s [%d]", msg.topic(), msg.partition())
