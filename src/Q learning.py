import random
import tkinter as tk

# קביעת פרמטרים של הסביבה
GRID_SIZE = 10  # גודל הרשת (10x10)
START = (0, 0)  # מיקום התחלתי
GOAL = (9, 9)  # מיקום היעד
ACTIONS = ['up', 'down', 'left', 'right']  # פעולות אפשריות

# יצירת טבלת תגמולים עם מכשולים
rewards = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # כל תזוזה מקבלת תגמול שלילי

# מכשולים (ערכים של -100 כדי שלא ייבחרו)
obstacles = [(3, 3), (3, 4), (4, 3), (6, 6), (6, 7), (7, 6)]
for obs in obstacles:
    rewards[obs[0]][obs[1]] = -100

# היעד מקבל תגמול גבוה
rewards[GOAL[0]][GOAL[1]] = 100

# יצירת טבלת Q
Q_table = [[[0 for _ in range(len(ACTIONS))]
               for _ in range(GRID_SIZE)]
               for _ in range(GRID_SIZE)]

# פרמטרים של Q-Learning
alpha = 0.5  # שיעור למידה  משפיע על איך נעדכן את ה-Q-Table.
gamma = 0.9  # מקדם הנחה קובע עד כמה נחשיב את העתיד (הערכים העתידיים של פעולות).
epsilon = 0.1  # הסתברות לבחירת פעולה אקראית (exploration) # קובע כמה פעמים נחפש פעולות חדשות לעומת ניצול הפעולות הטובות ביותר לפי ה-Q-Table.


# פונקציה המחשבת את המיקום הבא
def get_next_position(position, action):
    x, y = position
    if action == 'up' and x > 0:
        x -= 1
    elif action == 'down' and x < GRID_SIZE - 1:
        x += 1
    elif action == 'left' and y > 0:
        y -= 1
    elif action == 'right' and y < GRID_SIZE - 1:
        y += 1
    return (x, y)


# אלגוריתם Q-Learning
for episode in range(1000):  # מספר פרקים לאימון
    position = START  # אתחול למיקום ההתחלתי של הרובוט
    print(f"פרק {episode + 1}: התחל מ-{position}")  # הדפסת המיקום ההתחלתי של הרובוט בכל פרק

    while position != GOAL:  # ריצה עד הגעה ליעד
        # בחירת פעולה (exploration/exploitation)
        if random.uniform(0, 1) < epsilon:  # החלטה אם לבצע חיפוש אקראי (exploration) או ניצול הידע הקיים (exploitation)
            action = random.choice(ACTIONS)  # אם בחרנו חיפוש אקראי, בוחרים פעולה באופן אקראי מתוך פעולות אפשריות
        else:
            # אם בחרנו ניצול, בוחרים את הפעולה עם הערך הגבוה ביותר ב-Q-Table עבור המיקום הנוכחי
            action = ACTIONS[max(range(len(ACTIONS)), key=lambda i: Q_table[position[0]][position[1]][i])]

        # חישוב המיקום הבא לאחר ביצוע הפעולה הנבחרת
        next_position = get_next_position(position, action)

        # אם המיקום הבא הוא מכשול (הערך שלו הוא -100), בוחרים פעולה אחרת
        if rewards[next_position[0]][next_position[1]] == -100:
            continue  # אם יש מכשול, החזרה להתחלת הלולאה וניסיון פעולה אחרת

        # עדכון Q-Value של המיקום הנוכחי והפעולה שנבחרה
        reward = rewards[next_position[0]][next_position[1]]  # קבלת הערך של תגמול מהמיקום הבא
        best_next_Q = max(Q_table[next_position[0]][next_position[1]])  # מציאת הערך המקסימלי ב-Q-Table של המיקום הבא
        action_index = ACTIONS.index(action)  # זיהוי האינדקס של הפעולה הנבחרת

        # הדפסת ערך ה-Q לפני העדכון
        print(f"  מ-{position} בחרנו פעולה {action} -> מיקום הבא {next_position}")
        print(f"  ערך Q קודם: {Q_table[position[0]][position[1]][action_index]}")

        # עדכון ערך ה-Q לפי נוסחת עדכון Q-Learning
        Q_table[position[0]][position[1]][action_index] += alpha * (
                reward + gamma * best_next_Q - Q_table[position[0]][position[1]][action_index])

        # הדפסת ערך Q לאחר העדכון
        print(f"  ערך Q חדש: {Q_table[position[0]][position[1]][action_index]}")
        print("")

        # עדכון המיקום הנוכחי להיות המיקום הבא
        position = next_position  # עדכון המיקום כדי להמשיך בפרק הבא



# מציאת המסלול הטוב ביותר
def find_path(start, goal):
    path = [start]
    position = start
    while position != goal:
        action = ACTIONS[max(range(len(ACTIONS)), key=lambda i: Q_table[position[0]][position[1]][i])]
        position = get_next_position(position, action)
        path.append(position)
    return path


# תוצאה: מציאת המסלול הטוב ביותר
best_path = find_path(START, GOAL)


# יצירת ממשק גרפי עם tkinter
class RobotGUI:
    def __init__(self, grid_size, path):
        self.grid_size = grid_size
        self.path = path
        self.current_step = 0
        self.path_coordinates = []  # רשימה שתכיל את כל נקודות המסלול

        # יצירת חלון
        self.window = tk.Tk()
        self.window.title("ניווט רובוט עם מכשולים ו-Q-Learning")

        # יצירת קנבס לציור הרשת
        self.canvas = tk.Canvas(self.window, width=500, height=500)
        self.canvas.pack()

        # גודל תא
        self.cell_size = 500 // self.grid_size

        # ציור רשת
        self.draw_grid()

        # התחלת תנועת הרובוט
        self.robot = self.canvas.create_oval(5, 5, self.cell_size - 5, self.cell_size - 5, fill="blue")
        self.window.after(100, self.move_robot)

        self.window.mainloop()

    def draw_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                color = "white"

                if (i, j) == GOAL:
                    color = "green"
                elif (i, j) == START:
                    color = "yellow"
                elif (i, j) in obstacles:
                    color = "red"  # מכשול צבע אדום

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

    def move_robot(self):
        if self.current_step < len(self.path):
            # חישוב מיקום הרובוט
            x, y = self.path[self.current_step]
            x1 = y * self.cell_size + 5
            y1 = x * self.cell_size + 5
            x2 = x1 + self.cell_size - 10
            y2 = y1 + self.cell_size - 10

            # עדכון מיקום הרובוט
            self.canvas.coords(self.robot, x1, y1, x2, y2)

            # הוספת הנקודה למסלול
            self.path_coordinates.append(self.path[self.current_step])

            self.current_step += 1
            self.window.after(500, self.move_robot)  # עיכוב של 500ms בין תנועה לתנועה

        if self.current_step == len(self.path):
            # הדפסת מסלול אופטימלי בסוף
            print("מסלול אופטימלי: ")
            for coord in self.path_coordinates:
                print(coord)


# הרצת הממשק הגרפי
RobotGUI(GRID_SIZE, best_path)
