from unittest.mock import patch, MagicMock
import pytest
from engine.outreach import send_email


def test_send_email_simulated():
    res = send_email("candidate@example.com", "Test Subject", "Test Body")
    assert res["success"] is True
    assert res["mode"] == "Simulated"


def test_send_email_invalid_recipient():
    res = send_email("", "Test Subject", "Test Body")
    assert res["success"] is False
    assert "recipient email" in res["error"]

    res_none = send_email("Not Provided", "Test Subject", "Test Body")
    assert res_none["success"] is False


def test_send_email_smtp_incomplete_config():
    config = {"mode": "SMTP", "host": "smtp.example.com", "port": 587}
    res = send_email("cand@example.com", "Subj", "Body", smtp_config=config)
    assert res["success"] is False
    assert "Authentication" in res["error"]


def test_send_email_smtp_tls_success():
    config = {
        "mode": "SMTP",
        "host": "smtp.gmail.com",
        "port": 587,
        "user": "test@gmail.com",
        "password": "password",
        "sender_name": "Test Team"
    }
    
    mock_smtp_instance = MagicMock()
    mock_smtp_instance.__enter__.return_value = mock_smtp_instance
    with patch("smtplib.SMTP", return_value=mock_smtp_instance) as mock_smtp_cls:
        res = send_email("recipient@example.com", "Subject", "Body", smtp_config=config)
        assert res["success"] is True
        assert res["mode"] == "SMTP"
        mock_smtp_cls.assert_called_once_with("smtp.gmail.com", 587, timeout=10)
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("test@gmail.com", "password")
        mock_smtp_instance.sendmail.assert_called_once()


def test_send_email_smtp_ssl_success():
    config = {
        "mode": "SMTP",
        "host": "smtp.gmail.com",
        "port": 465,
        "user": "test@gmail.com",
        "password": "password",
        "sender_name": "Test Team"
    }
    
    mock_smtp_ssl_instance = MagicMock()
    mock_smtp_ssl_instance.__enter__.return_value = mock_smtp_ssl_instance
    with patch("smtplib.SMTP_SSL", return_value=mock_smtp_ssl_instance) as mock_smtp_ssl_cls:
        res = send_email("recipient@example.com", "Subject", "Body", smtp_config=config)
        assert res["success"] is True
        assert res["mode"] == "SMTP"
        mock_smtp_ssl_cls.assert_called_once()
        mock_smtp_ssl_instance.login.assert_called_once_with("test@gmail.com", "password")
        mock_smtp_ssl_instance.sendmail.assert_called_once()


def test_send_email_smtp_failure():
    config = {
        "mode": "SMTP",
        "host": "smtp.gmail.com",
        "port": 587,
        "user": "test@gmail.com",
        "password": "password",
        "sender_name": "Test Team"
    }
    
    with patch("smtplib.SMTP", side_effect=Exception("Connection refused")):
        res = send_email("recipient@example.com", "Subject", "Body", smtp_config=config)
        assert res["success"] is False
        assert "Connection refused" in res["error"]
