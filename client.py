import time, socket, sys
import os
import keyboard

scale = 1


def what_to_find(): #check what are we searching for
    f = open(get_download_path() + '\search.txt', "r")
    file_conatins = f.read()
    print(file_conatins)
    f.close()
    os.remove(get_download_path() + '\search.txt')

    if "ez" in file_conatins:
        return (-2)
    else:
        if "traffic light" in file_conatins:
            return (7)
        else:
            if "palm tree" in file_conatins:
                return (6)
            else:
                if "motorcycle" in file_conatins:
                    return (1)
                else:
                    if "fire hydrant" in file_conatins:
                        return (4)
                    else:
                        if "car" in file_conatins or "vehicle" in file_conatins:
                            return (3)
                        else:
                            if "bus" in file_conatins:
                                return (2)
                            else:
                                if "bicycle" in file_conatins:
                                    return (1)
                                else:
                                    return (-1)

def get_download_path():
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'downloads')

def get_text(file_name):
    with open(get_download_path() + file_name, 'r') as f:
        return f.read()


print("Initialising....\n")
time.sleep(1)

s = socket.socket()
shost = socket.gethostname()
ip = socket.gethostbyname(shost)
print(shost, "(", ip, ")\n")
host = "192.168.1.11"
name = "stav"
port = 1234
print("\nTrying to connect to ", host, "(", port, ")\n")
time.sleep(1)
s.connect((host, port))
print("Connected...\n")



while True:
    if os.path.isfile(get_download_path() + '/search.txt'):  # checks if there is what to search for
        message = what_to_find()
        s.send((str(message)).encode())

    if os.path.isfile(get_download_path() + '/img_url.txt'):
        message = get_text("\img_url.txt")
        print(message)
        s.send(message.encode())
        os.remove(get_download_path() + '\img_url.txt')
        message = s.recv(1024)
        message = message.decode()
        print(message)
        array_of_images = message
        array_of_images=array_of_images.split(", ")
        array_of_images[len(array_of_images)- 1]=array_of_images[len(array_of_images)- 1][0]
        array_of_images[0] = array_of_images[0][1]
        print(array_of_images)
        print(len(array_of_images))
        for i in range(len(array_of_images)):
            keyboard.press_and_release('left_arrow')

            if int(array_of_images[i]) == 1:
                keyboard.press_and_release('enter')

        for i in range(int(len(array_of_images) ** 0.5)):
            keyboard.press_and_release('down_arrow')
            #keyboard.press_and_release('enter')


    #if message == "[e]":
        #message = "Left chat room!"
        #s.send(message.encode())
        #print("\n")
        #break
    #s.send(message.encode())