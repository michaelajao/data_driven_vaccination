# from datetime import datetime, timedelta
# from ics import Calendar, Event, Alarm

# # Create a new calendar
# c = Calendar()

# # Helper function to create events easily
# def create_event(summary, description, start_time, duration_hours=0, duration_minutes=0):
#     e = Event()
#     e.name = summary
#     e.description = description
#     e.begin = start_time
#     e.duration = timedelta(hours=duration_hours, minutes=duration_minutes)
#     # Adding a reminder 10 minutes before the event starts
#     e.alarms.append(Alarm(trigger=timedelta(minutes=-10)))
#     c.events.add(e)

# # Specify the date for the schedule
# date_str = "2024-04-05"  # Adjust the date as needed

# # Morning Exercise
# create_event(
#     "Morning Exercise",
#     "30 minutes of low-impact activities, considering your knee. A mix of cycling, swimming, or yoga could be beneficial.",
#     f"{date_str} 07:15:00",
#     duration_minutes=30
# )

# # Research and Reading
# create_event(
#     "Research and Reading",
#     "Dedicate this time to reading research papers. Your mind is fresh, making it a great time for this task.",
#     f"{date_str} 09:00:00",
#     duration_hours=2
# )

# # Brunch
# create_event(
#     "Brunch",
#     "Enjoy a hearty brunch to recharge.",
#     f"{date_str} 12:00:00",
#     duration_hours=1
# )

# # Online Classes/Meetings
# create_event(
#     "Online Classes",
#     "Engage with your classes or meetings, a prime time for interactive learning.",
#     f"{date_str} 13:00:00",
#     duration_hours=2
# )

# # Hackerrank Practice
# create_event(
#     "Hackerrank Practice",
#     "Spend 30 minutes on a HackerRank question to sharpen your problem-solving skills.",
#     f"{date_str} 15:15:00",
#     duration_minutes=30
# )

# # Coding
# create_event(
#     "Coding",
#     "Dive into your coding projects. This block includes a focused coding session before and after your online classes.",
#     f"{date_str} 11:00:00",
#     duration_hours=3
# )

# # Coding Continuation
# create_event(
#     "Coding Continuation",
#     "Continue with your coding projects.",
#     f"{date_str} 15:45:00",
#     duration_hours=2,  # Adjusted for additional coding time post-HackerRank
# )

# # Dinner
# create_event(
#     "Dinner",
#     "Enjoy your dinner. Maybe watch an episode of your TV series.",
#     f"{date_str} 20:00:00",
#     duration_hours=1
# )

# # Leisure Time
# create_event(
#     "Leisure Time",
#     "Talk to your girlfriend and watch a TV series together.",
#     f"{date_str} 21:00:00",
#     duration_hours=3
# )

# # Exporting to a file
# with open('my_complete_schedule.ics', 'w') as my_file:
#     my_file.writelines(c)

# print("Complete schedule created successfully!")


from datetime import datetime, timedelta
from ics import Calendar, Event, DisplayAlarm

# Initialize a new Calendar
c = Calendar()


def create_event(
    calendar, name, description, start, duration_hours=0, duration_minutes=0
):
    event = Event()
    event.name = name
    event.description = description
    event.begin = start.strftime("%Y-%m-%d %H:%M:%S")
    event.duration = timedelta(hours=duration_hours, minutes=duration_minutes)
    # Add a reminder (alarm) 10 minutes before the event starts
    # reminder = DisplayAlarm(trigger=timedelta(minutes=-10), display_text="Reminder")
    # event.alarms.append(reminder)
    calendar.events.add(event)


# Define the start date
start_date = datetime(2024, 4, 4)  # Example date, adjust as necessary

# Schedule events
events = [
    (
        "Wake up",
        "Start your day with some light stretching to gently wake your body up.",
        start_date.replace(hour=7, minute=0),
        0,
        15,
    ),
    (
        "Exercise",
        "30 minutes of low-impact activities, considering your knee.",
        start_date.replace(hour=7, minute=15),
        0,
        30,
    ),
    (
        "Shower and get ready",
        "Time to freshen up and prepare for the day.",
        start_date.replace(hour=7, minute=45),
        0,
        45,
    ),
    (
        "Breakfast",
        "Start your day with a nutritious meal to fuel your morning.",
        start_date.replace(hour=8, minute=30),
        0,
        30,
    ),
    (
        "Research and Reading",
        "Dedicate this time to reading research papers.",
        start_date.replace(hour=9, minute=0),
        2,
        0,
    ),
    (
        "Coding",
        "Dive into your coding projects before your class.",
        start_date.replace(hour=11, minute=0),
        1,
        0,
    ),
    (
        "Brunch",
        "Enjoy a hearty brunch to recharge.",
        start_date.replace(hour=12, minute=0),
        1,
        0,
    ),
    (
        "Online Classes/Meetings",
        "Engage with your classes or meetings.",
        start_date.replace(hour=13, minute=0),
        2,
        0,
    ),
    (
        "Hackerrank Practice",
        "Spend 30 minutes on a HackerRank question.",
        start_date.replace(hour=15, minute=15),
        0,
        30,
    ),
    (
        "Free Time/Short Break",
        "Take a brief break to relax and reset.",
        start_date.replace(hour=15, minute=45),
        0,
        15,
    ),
    (
        "Coding",
        "Continue with your coding projects.",
        start_date.replace(hour=16, minute=0),
        3,
        0,
    ),
    (
        "Wrap up & Relax",
        "Start to wind down your work and transition into relaxation mode.",
        start_date.replace(hour=19, minute=0),
        1,
        0,
    ),
    ("Dinner", "Enjoy your dinner.", start_date.replace(hour=20, minute=0), 1, 0),
    (
        "Leisure Time",
        "Begin your leisure time, including talking to your girlfriend and watching a TV series together.",
        start_date.replace(hour=21, minute=0),
        2,
        0,
    ),
    (
        "Wind Down",
        "Start to get ready for bed. Consider some light reading or meditation.",
        start_date.replace(hour=23, minute=0),
        1,
        0,
    ),
    (
        "Bedtime",
        "Time to sleep and recharge for the next day.",
        start_date.replace(day=6, hour=0, minute=0),
        0,
        0,
    ),
]

for name, description, start, hours, minutes in events:
    create_event(c, name, description, start, hours, minutes)

# Saving the calendar to a file
with open("full_daily_schedule.ics", "w") as file:
    file.writelines(c)

print("Your full daily schedule with reminders has been created successfully!")
