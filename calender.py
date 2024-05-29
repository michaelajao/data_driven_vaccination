from datetime import datetime, timedelta
from ics import Calendar, Event, DisplayAlarm

# Initialize a new Calendar
c = Calendar()


# Helper function to create events easily
def create_event(
    calendar, name, description, start, duration_hours=0, duration_minutes=0
):
    event = Event()
    event.name = name
    event.description = description
    event.begin = start.strftime("%Y-%m-%d %H:%M:%S")
    event.duration = timedelta(hours=duration_hours, minutes=duration_minutes)
    # Add a reminder (alarm) 10 minutes before the event starts
    reminder = DisplayAlarm(trigger=timedelta(minutes=-10), display_text="Reminder")
    event.alarms.append(reminder)
    calendar.events.add(event)


# Define the start date
start_date = datetime(2024, 5, 30)  # Example date, adjust as necessary

# Schedule events
events = [
    (
        "Wake up, Pray and Exercise",
        "Start your day with some light stretching to gently wake your body up, followed by a short prayer and 30 minutes of low-impact activities, considering your knee.",
        start_date.replace(hour=6, minute=0),
        0,
        30,
    ),
    (
        "Shower and Get Ready",
        "Time to freshen up and prepare for the day.",
        start_date.replace(hour=6, minute=30),
        0,
        30,
    ),
    (
        "Breakfast and coffee",
        "Start your day with a nutritious meal to fuel your morning.",
        start_date.replace(hour=7, minute=0),
        0,
        30,
    ),
    (
        "Research and Reading",
        "Dedicate this time to reading research papers.",
        start_date.replace(hour=8, minute=0),
        2,
        0,
    ),
    (
        "Coding",
        "Dive into your coding projects for your research work.",
        start_date.replace(hour=10, minute=0),
        2,
        0,
    ),
    (
        "Brunch",
        "Enjoy a hearty brunch to recharge.",
        start_date.replace(hour=14, minute=0),
        0,
        30,
    ),
    (
        "Hackerrank Practice",
        "Spend 30 minutes on a HackerRank question.",
        start_date.replace(hour=14, minute=30),
        0,
        30,
    ),
    (
        "Free Time/Short Break",
        "Take a brief break to relax and reset.",
        start_date.replace(hour=15, minute=0),
        0,
        30,
    ),
    (
        "Research paper writing",
        "Work on writing the research paper.",
        start_date.replace(hour=15, minute=30),
        2,
        0,
    ),
    (
        "Bike to the gym and exercise the legs",
        "1 hour of strength training and cardio exercises.",
        start_date.replace(hour=17, minute=30),
        1,
        0,
    ),
    (
        "Bike ride back home",
        "Enjoy a bike ride back home and listen to a podcast on the way.",
        start_date.replace(hour=18, minute=30),
        0,
        15,
    ),
    (
        "Shower, relax and unwind",
        "Take a shower and relax after your workout.",
        start_date.replace(hour=19, minute=0),
        0,
        30,
    ),
    (
        "Dinner",
        "Enjoy your dinner. Maybe watch an episode of your TV series.",
        start_date.replace(hour=19, minute=30),
        0,
        30,
    ),
    (
        "Quality time with family",
        "Spend quality time with your family, playing games or watching a movie and including talking to your girlfriend and watching a TV series together.",
        start_date.replace(hour=20, minute=0),
        1,
        30,
    ),
    (
        "Work on the presentation for tomorrow meeting with my supervisor to discuss the progress of the research work",
        "Prepare for the meeting with your supervisor.",
        start_date.replace(hour=21, minute=30),
        2,
        0,
    ),
    (
        "Prepare for bed",
        "Wind down and prepare for a restful night's sleep.",
        start_date.replace(hour=23, minute=30),
        0,
        30,
    ),
]

# Create events from the list
for name, description, start, hours, minutes in events:
    create_event(c, name, description, start, hours, minutes)

# Saving the calendar to a file
with open("full_daily_schedule.ics", "w") as file:
    file.writelines(c)

print("Your full daily schedule with reminders has been created successfully!")
