import threading


class APIMetrics:
    """Track basic API request statistics safely across threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total_requests = 0
        self._successful_predictions = 0
        self._failed_requests = 0
        self._total_response_time_ms = 0.0

    def record_request(
        self,
        status_code: int,
        response_time_ms: float,
    ) -> None:
        """Record one completed HTTP request."""

        with self._lock:
            self._total_requests += 1

            if status_code >= 400:
                self._failed_requests += 1

            self._total_response_time_ms += response_time_ms

    def record_successful_prediction(self) -> None:
        """Record one successful prediction."""

        with self._lock:
            self._successful_predictions += 1

    def snapshot(self) -> dict:
        """Return the current API metrics."""

        with self._lock:
            if self._total_requests == 0:
                average_response_time_ms = 0.0
            else:
                average_response_time_ms = (
                    self._total_response_time_ms
                    / self._total_requests
                )

            return {
                "total_requests": self._total_requests,
                "successful_predictions": (
                    self._successful_predictions
                ),
                "failed_requests": self._failed_requests,
                "average_response_time_ms": round(
                    average_response_time_ms,
                    3,
                ),
            }