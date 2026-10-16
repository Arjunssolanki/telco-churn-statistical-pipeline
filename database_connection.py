import os
import mysql.connector
from contextlib import contextmanager

def load_environment_variables(filepath=".env"):
    """Manually parses local environment lines if package bindings aren't present."""
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

# Initialize credential environment array
load_environment_variables()

@contextmanager
def get_mysql_connection():
    """Provides a safe connection resource wrapper for database queries."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306)),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "")
        )
        yield connection
    except mysql.connector.Error as err:
        print(f"🛑 Connection layer exception encountered: {err}")
        raise err
    finally:
        if connection and connection.is_connected():
            connection.close()
