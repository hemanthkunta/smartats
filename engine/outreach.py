"""
outreach.py — manages email outreach dispatch (Simulated and SMTP Direct Send).
"""
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(to_address, subject, body, smtp_config=None):
    """
    Sends an email to a single recipient.
    Supports secure SMTP (SSL/TLS) or simulated sandboxed sending.
    """
    if not to_address or to_address == "Not Provided":
        return {"success": False, "error": "Invalid or missing recipient email address."}

    if smtp_config is None:
        smtp_config = {"mode": "Simulated"}

    if smtp_config.get("mode") == "Simulated":
        print(f"Simulated email sent successfully to {to_address}. Subject: {subject}")
        return {
            "success": True,
            "mode": "Simulated",
            "info": f"Simulated email sent successfully to {to_address}."
        }

    # Direct SMTP Live Server Mode
    host = smtp_config.get("host", "smtp.gmail.com")
    port = int(smtp_config.get("port", 587))
    user = smtp_config.get("user", "")
    password = smtp_config.get("password", "")
    sender_name = smtp_config.get("sender_name", "SmartATS Evaluator")

    if not user or not password:
        return {"success": False, "error": "SMTP Authentication details (username/password) are incomplete."}

    # Setup email message
    msg = MIMEMultipart()
    msg["From"] = f"{sender_name} <{user}>"
    msg["To"] = to_address
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        # Standard SSL Port (465)
        if port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=context, timeout=10) as server:
                server.login(user, password)
                server.sendmail(user, to_address, msg.as_string())
        # Standard TLS Port (587 or others)
        else:
            context = ssl.create_default_context()
            with smtplib.SMTP(host, port, timeout=10) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(user, password)
                server.sendmail(user, to_address, msg.as_string())
        
        return {"success": True, "mode": "SMTP"}
    except Exception as e:
        return {"success": False, "error": str(e)}
