class QuizBrain:

    def __init__(self, question_list):
        self.question_no = 0
        self.question_list = question_list
        self.score = 0

    def still_has_questions(self):
        return self.question_no < len(self.question_list)

    def next_question(self):
        question = self.question_list[self.question_no]
        self.question_no += 1
        user_input = (input(f'Q.{self.question_no}: {question.text} (True / False)\n')).lower()
        self.check_answer(user_input, question.answer)

    def check_answer(self, user_input, correct_answer):
        if user_input != correct_answer.lower():
            print('You got it wrong!')
        else:
            self.score += 1
            print('You got it right!')

        print(f'The correct answer is: {correct_answer}')
        print(f'Your current score is: {self.score} / {self.question_no}')
        print('\n')

