import Task_4
import Task_5
import Task_6
import Task_7


def main():
    task_number = int(input('Please choose from Task 4, 5, 6, 7: '))
    if task_number == 4:
        Task_4.main()
    elif task_number == 5:
        Task_5.main()
    elif task_number == 6:
        Task_6.main()
    else:
        Task_7.main()


main()
