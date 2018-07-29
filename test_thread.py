import threading

# def main():
#     print(threading.active_count())
#     print(threading.enumerate())
#     print(threading.current_thread())


def thread_job():
    print('This is a thread of %s' % threading.current_thread())

def main():
    thread = threading.Thread(target=thread_job,)   # 定义线程
    thread.start()  # 让线程开始工作
    print(threading.active_count())
    print(threading.current_thread())

if __name__ == '__main__':
    main()
