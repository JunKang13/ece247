Quiz = []
Answers = []
QuizQuestions = []


# recursive function to add up to i
def add_up_to(i):
    if i == 0:
        return 0
    return i + add_up_to(i - 1)


def add_questions():
    question = input("Enter Question: ")
    Quiz.append(question)
    flag = True
    while flag:
        ContinueAdding = input("Would you like to add another question to the quiz?: ")
        if ContinueAdding == "yes":
            add_questions()
        elif ContinueAdding == "no":
            for line in Quiz:
                quiz_file.write(line + "\n")
            quiz_file.close()
            flag = False
        else:
            print("Type it again")


def append_answers(indent=0):
    if indent == len(QuizQuestions):
        print("Quiz finished, answers saved. You can take another quiz.")
        return ""
    question = QuizQuestions[indent]
    print(QuizQuestions[indent])
    answer = input(" ")
    answers_to_question = str(Answer(question, answer))
    # print(answers_to_question)
    Answers.append(answers_to_question)
    indent += 1
    append_answers(indent)


flag = True
while flag:
    WHO = input("Are you a Professor Or A Student?: (exit to quit)\n")
    if WHO == "Professor":
        while True:
            Questions = input("Would you like to add questions to a quiz or create a new quiz?: ")
            if Questions == "yes":
                QuizName = input("Enter the name of the quiz: ")
                filename1 = f"{QuizName}.txt"
                try:
                    quiz_file = open(filename1, "a")
                except FileNotFoundError as e:
                    quiz_file = open(filename1, "w+")
                Quiz.clear()
                add_questions()

            elif Questions == "no":
                flag = False
                break
            else:
                print("Type it again")

    elif WHO == "Student":

        # exception
        while True:
            take_quiz = input("Would you like to take a quiz? yes or no\n")
            if take_quiz == "yes":

                while True:
                    quiz_name = input("Enter the name of the quiz: ")
                    filename1 = f"{quiz_name}.txt"
                    try:
                        quiz_file = open(filename1, "r")
                        break
                    except Exception as e:
                        print(e)

                QuizQuestions.clear()
                Answers.clear()
                for line in quiz_file.readlines():
                    QuizQuestions.append(line.rstrip())
                # print(QuizQuestions)

                studentName = input("Your name: ")


                class Answer:
                    def __init__(self, questions, answers):
                        self.questions = questions
                        self.answers = answers

                    def __str__(self):
                        return f"Q: {self.questions} A: {self.answers}"


                # recursion

                append_answers()

                filename2 = f"{quiz_name}-{studentName}.txt"

                # exception
                try:
                    student_file = open(filename2, "r+")
                except FileNotFoundError as e:
                    student_file = open(filename2, "w+")

                for line in Answers:
                    student_file.write(line + "\n")

                student_file.close()
            elif take_quiz == "no":
                break
            else:
                print("Type it again")

    elif WHO == "exit":
        flag = False
        break
    else:
        print("Type it again")
