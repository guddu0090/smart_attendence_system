import sqlite3
from datetime import datetime

def create_database_with_sample_data():
    """
    This script creates the 'attendance.db' file with the necessary tables ('students' and 'attendance')
    and populates them with some sample data for testing.
    """
    db_file = 'attendance.db'
    try:
        # Connect to the database. If the file doesn't exist, it will be created.
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        print("Database connection established.")

        # --- Table 1: students ---
        # Stores the information for each registered student.
        print("Creating 'students' table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            UNIQUE(name, class) -- Ensures no duplicate students in the same class
        )
        """)
        print("'students' table created or already exists.")

        # --- Table 2: attendance ---
        # Stores the attendance log for each student on a specific date.
        print("Creating 'attendance' table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            time TEXT NOT NULL,
            date DATE NOT NULL,
            UNIQUE(name, date) -- A student can only be marked present once per day
        )
        """)
        print("'attendance' table created or already exists.")

        # --- Insert Sample Data ---
        # Add some sample students to the roster.
        try:
            print("Inserting sample students...")
            sample_students = [
                ('Alice Johnson', 'Class 10A'),
                ('Bob Williams', 'Class 10A'),
                ('Charlie Brown', 'Class 10B')
            ]
            cursor.executemany("INSERT INTO students (name, class) VALUES (?, ?)", sample_students)
            print(" -> 3 sample students inserted.")
        except sqlite3.IntegrityError:
            print(" -> Sample students already exist in the 'students' table.")

        # Add some sample attendance records for today.
        try:
            print("Inserting sample attendance records...")
            today_date = datetime.now().strftime("%Y-%m-%d")
            sample_attendance = [
                ('Alice Johnson', '09:05:12', today_date),
                ('Charlie Brown', '09:03:45', today_date)
            ]
            cursor.executemany("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", sample_attendance)
            print(f" -> 2 sample attendance records for {today_date} inserted.")
        except sqlite3.IntegrityError:
            print(f" -> Sample attendance records for {today_date} already exist.")

        # Commit all changes to the database
        conn.commit()
        print("All changes have been committed to the database.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the connection
        if conn:
            conn.close()
            print(f"Database '{db_file}' is set up and connection is closed.")

if __name__ == '__main__':
    create_database_with_sample_data()
