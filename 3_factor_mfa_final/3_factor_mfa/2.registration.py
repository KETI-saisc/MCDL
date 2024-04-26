# -*- coding:utf-8 -*-

##___________________________________________Camera____________________________________________
#import cv2
#import numpy as np
#import os

#___________________________________________Finger____________________________________________
import serial
import time
import threading
import sys
import RPi.GPIO as GPIO

##__________________________________Face__________________________________
#id = 0
#confidence = 0
#recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read('/home/jihoon/Desktop/jaxson/FacialRecognitionProject/trainer/trainer.yml')
#cascadePath = "/home/jihoon/Desktop/jaxson/face_frame.xml"
#faceCascade = cv2.CascadeClassifier(cascadePath)
#font = cv2.FONT_HERSHEY_SIMPLEX
## iniciate id counter
#id = 1
## names related to ids: example ==> Marcelo: id=1,  etc
#names = ['None', 'face detected', 'Paula', 'Ilza', 'Z', 'W']
## Initialize and start realtime video capture
#cam = cv2.VideoCapture(0)
#cam.set(3, 640)  # set video widht
#cam.set(4, 480)  # set video height
## Define min window size to be recognized as a face
#minW = 0.1 * cam.get(3)
#minH = 0.1 * cam.get(4)


#__________________________________Finger__________________________________
TRUE         =  1
FALSE        =  0

# Basic response message definition
ACK_SUCCESS           = 0x00
ACK_FAIL              = 0x01
ACK_FULL              = 0x04
ACK_NO_USER           = 0x05
ACK_TIMEOUT           = 0x08
ACK_GO_OUT            = 0x0F     # The center of the fingerprint is out of alignment with sensor

# User information definition
ACK_ALL_USER          = 0x00
ACK_GUEST_USER        = 0x01
ACK_NORMAL_USER       = 0x02
ACK_MASTER_USER       = 0x03

USER_MAX_CNT          = 1000        # Maximum fingerprint number

# Command definition
CMD_HEAD              = 0xF5
CMD_TAIL              = 0xF5
CMD_ADD_1             = 0x01
CMD_ADD_2             = 0x02
CMD_ADD_3             = 0x03
CMD_MATCH             = 0x0C
CMD_DEL               = 0x04
CMD_DEL_ALL           = 0x05
CMD_USER_CNT          = 0x09
CMD_COM_LEV           = 0x28
CMD_LP_MODE           = 0x2C
CMD_TIMEOUT           = 0x2E

CMD_FINGER_DETECTED   = 0x14



Finger_WAKE_Pin   = 23
Finger_RST_Pin    = 24

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(Finger_WAKE_Pin, GPIO.IN)
GPIO.setup(Finger_RST_Pin, GPIO.OUT)
GPIO.setup(Finger_RST_Pin, GPIO.OUT, initial=GPIO.HIGH)

g_rx_buf            = []
PC_Command_RxBuf    = []
Finger_SleepFlag    = 0

# door motor setting
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(12,GPIO.OUT)
GPIO.setup(13,GPIO.OUT)

#rLock = threading.RLock()
ser = serial.Serial("/dev/ttyTHS1", 19200)


def Door_Open_Control(enable):
    # stop motor
    GPIO.output(12,GPIO.LOW)
    GPIO.output(13,GPIO.LOW)
    time.sleep(.2)

    if enable == "1":
        # door open
        GPIO.output(13,GPIO.LOW)
        GPIO.output(12,GPIO.HIGH)
        time.sleep(3)

    else:
        # door close
        GPIO.output(12,GPIO.LOW)
        GPIO.output(13,GPIO.HIGH)
        time.sleep(3)

    # stop motor
    GPIO.output(12,GPIO.LOW)
    GPIO.output(13,GPIO.LOW)
    time.sleep(.2)


##__________________________________Face__________________________________
#
#global cnt_correct
#cnt_correct = 0
#
#Door_Open_Control("0")
#
#while True:
#    ret, img = cam.read()
#    img = cv2.flip(img, -1)  # Flip vertically
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#    faces = faceCascade.detectMultiScale(
#        gray,
#        scaleFactor=1.2,
#        minNeighbors=5,
#        minSize=(int(minW), int(minH)),
#    )
#    for (x, y, w, h) in faces:
#        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
#
#        # Check if confidence is less them 100 ==> "0" is perfect match
#        if (confidence < 100):
#            id = names[id]
#            confidence_str = "  {0}%".format(round(100 - confidence))
#        else:
#            id = "unknown"
#            confidence_str = "  {0}%".format(round(100 - confidence))
#
#        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
#        # cv2.putText(img, str(confidence_str), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
#        cv2.putText(img, "Checking...", (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
#
#
#    # print (id)
#    # print (confidence)
#    cv2.imshow('camera', img)
#    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
#    if k == 27:
#        break
#    print(100-confidence)
#    if (id == 'face detected') and ((100-confidence) > 40):
#        cnt_correct += 1
#        if cnt_correct > 50:
#            print("________________________________________________________________________________")
#            print("___________________________Face Authentication Success__________________________")
#            print("________________________________________________________________________________")
#            break
## Do a bit of cleanup
#print("\n [INFO] Exiting Program and cleanup stuff")
#cam.release()
#cv2.destroyAllWindows()


#__________________________________Finger__________________________________


#***************************************************************************
# @brief    send a command, and wait for the response of module
#***************************************************************************/
def  TxAndRxCmd(command_buf, rx_bytes_need, timeout):
    #print (command_buf) # 40,0,5,0,0
    #print (type(command_buf)) # list
    global g_rx_buf
    CheckSum = 0
    tx_buf = []
    tx = ""
    if command_buf == None:
        pass
    else:
        tx_buf.append(CMD_HEAD)
        for byte in command_buf:
            tx_buf.append(byte)
            CheckSum ^= byte

        tx_buf.append(CheckSum)
        tx_buf.append(CMD_TAIL)

# python 2
#        for i in tx_buf:
#            tx += chr(i)
#        ser.flushInput()
#        ser.write(tx)


# python3
        TxData = ser.write(bytes(tx_buf)) #.decode('hex'))
        print ("Tx :", tx_buf)
    g_rx_buf = []
    time_before = time.time()
    time_after = time.time()

    while time_after - time_before < timeout and len(g_rx_buf) < rx_bytes_need:  # Waiting for response
        bytes_can_recv = ser.inWaiting()
        if bytes_can_recv != 0:
            #g_rx_buf_utf += ser.read(bytes_can_recv).decode('utf-8')
            g_rx_buf += ser.read(bytes_can_recv)
            #g_rx_buf = g_rx_buf.decode('hex')
        time_after = time.time()

    # print ("rx: ", g_rx_buf)

#    for i in range(len(g_rx_buf)):
#        g_rx_buf[i] = ord(g_rx_buf[i])

    if len(g_rx_buf) != rx_bytes_need:
        return ACK_TIMEOUT
    if g_rx_buf[0] != CMD_HEAD:
        return ACK_FAIL
    if g_rx_buf[rx_bytes_need - 1] != CMD_TAIL:
        return ACK_FAIL
    if g_rx_buf[1] != tx_buf[1]:
        return ACK_FAIL
    print ("Rx :", g_rx_buf)

    #print (g_rx_buf[2:4]) #jiseong
    # if g_rx_buf[2:4] == # 하이로우 len이면
    #
    #     g_rx_buf += ser.read(bytes_can_recv)

    CheckSum = 0
    for index, byte in enumerate(g_rx_buf):
        if index == 0:
            continue
        if index == 6:
            if CheckSum != byte:
                return ACK_FAIL
        CheckSum ^= byte
    return  ACK_SUCCESS

#***************************************************************************
# @brief    Get Compare Level
#***************************************************************************/
def GetCompareLevel():
    global g_rx_buf
    command_buf = [CMD_COM_LEV, 0, 0, 1, 0]
    r = TxAndRxCmd(command_buf, 8, 0.1)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return g_rx_buf[3]
    else:
        return 0xFF

#***************************************************************************
# @brief    Set Compare Level,the default value is 5,
#           can be set to 0-9, the bigger, the stricter
#***************************************************************************/
def SetCompareLevel(level):
    global g_rx_buf
    command_buf = [CMD_COM_LEV, 0, level, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 0.1)

    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return  g_rx_buf[3]
    else:
        return 0xFF

#***************************************************************************
# @brief   Query the number of existing fingerprints
#***************************************************************************/
def GetUserCount():
    global g_rx_buf
    command_buf = [CMD_USER_CNT, 0, 0, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 0.1)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return g_rx_buf[3]
    else:
        return 0xFF
#***************************************************************************
# @brief   Get the time that fingerprint collection wait timeout
#***************************************************************************/
def GetTimeOut():
    global g_rx_buf
    command_buf = [CMD_TIMEOUT, 0, 0, 1, 0]
    r = TxAndRxCmd(command_buf, 8, 10)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return g_rx_buf[3]
    else:
        return 0xFF

#***************************************************************************
# @brief   Get image data (Jiseong)
#***************************************************************************/
def RequestImage():
    global g_rx_buf
    command_buf = [0x24, 0, 0, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 60)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        #print ("g_rx_buf: ", g_rx_buf)
        #print ("Hi(Len): ", g_rx_buf[2])
        #print ("Low(Len): ", g_rx_buf[3])
        return ACK_SUCCESS
    else:
        return 0xFF

#***************************************************************************
# @brief    Register fingerprint
#***************************************************************************/
def AddUser():
    global g_rx_buf
    r = GetUserCount()
    if r >= USER_MAX_CNT:
        return ACK_FULL

    command_buf = [CMD_ADD_1, 0, r+1, 3, 0]
    print(g_rx_buf)
    r = TxAndRxCmd(command_buf, 8, 6)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        command_buf[0] = CMD_ADD_3
        r = TxAndRxCmd(command_buf, 8, 6)
        print(g_rx_buf)
        if r == ACK_TIMEOUT:
            return ACK_TIMEOUT
        if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
            return ACK_SUCCESS
        else:
            return ACK_FAIL
    else:
        return ACK_FAIL


#***************************************************************************
# @brief    Clear fingerprints
#***************************************************************************/
def ClearAllUser():
    global g_rx_buf
    command_buf = [CMD_DEL_ALL, 0, 0, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 5)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return ACK_SUCCESS
    else:
        return ACK_FAIL


def ClearOneUser(user_id_hi, user_id_lo):
    global g_rx_buf
    command_buf = [CMD_DEL, user_id_hi, user_id_lo, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 5)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return ACK_SUCCESS
    else:
        return ACK_FAIL


 #***************************************************************************
# @brief    Check if user ID is between 1 and 3
#***************************************************************************/
def IsMasterUser(user_id):
    if user_id == 1 or user_id == 2 or user_id == 3:
        return TRUE
    else:
        return FALSE

#***************************************************************************
# @brief    Fingerprint matching
#***************************************************************************/
def VerifyUser():
    global g_rx_buf
    global verify_user_id_int
    command_buf = [CMD_MATCH, 0, 0, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 5)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and IsMasterUser(g_rx_buf[4]) == TRUE:
        verify_user_id_int = int(str(g_rx_buf[2])+str(g_rx_buf[3]), 16)
        #print (user_id_int)
        return ACK_SUCCESS
    elif g_rx_buf[4] == ACK_NO_USER:
        return ACK_NO_USER
    elif g_rx_buf[4] == ACK_TIMEOUT:
        return ACK_TIMEOUT
    else:
        return ACK_GO_OUT   # The center of the fingerprint is out of alignment with sensor


#***************************************************************************
# @brief    Analysis the command from PC terminal
#***************************************************************************/
def Analysis_PC_Command(command):
    global Finger_SleepFlag, g_rx_buf

    if  command == "1" and Finger_SleepFlag != 1:
        print ("Number of fingerprints already available:  %d"  % GetUserCount())

    elif command == "2" and Finger_SleepFlag != 1:
        print ("Add fingerprint  (Put your finger on sensor until successfully/failed information returned) ")
        r = AddUser()
        if r == ACK_SUCCESS:
            print ("Fingerprint added successfully !")
        elif r == ACK_FAIL:
            print ("Failed: Please try to place the center of the fingerprint flat to sensor, or this fingerprint already exists !")
        elif r == ACK_FULL:
            print ("Failed: The fingerprint library is full !")

    elif command == "3" and Finger_SleepFlag != 1:
        print ("Waiting enter Fingerprint...")
        r = VerifyUser()
        if r == ACK_SUCCESS:
            print ("Matching successful !")
        elif r == ACK_NO_USER:
            print ("Failed: This fingerprint was not found in the library !")
        elif r == ACK_TIMEOUT:
            print ("Failed: Time out !")
        elif r == ACK_GO_OUT:
            print ("Failed: Please try to place the center of the fingerprint flat to sensor !")

    elif command == "4" and Finger_SleepFlag != 1:
        # 1. Delete all fingerprints
        ClearAllUser()
        # 2. Delete just a fingerprint
        # ClearOneUser()
        print ("All fingerprints have been cleared !")

    elif command[:2] == '4,' and Finger_SleepFlag != 1:
        command = command.split(',')
        command = command[-1]
        command = command.replace(' ','')
        if int(command) < 1000:
            command_hex = str(hex(int(command)))
            command_hex_list = command_hex.split('x')
            command_hex_str = command_hex_list[-1]

            print("Delete User ID :", command)
            if len(command_hex_str) <= 2:
                user_id_hi = 0
                user_id_lo = int(command_hex_str)
            elif len(command_hex_str) <= 4:
                user_id_hi = int(command_hex_str[:2])
                user_id_lo = int(command_hex_str[2:])
            else:
                print ("wow~! what a wonderful error")

            print ("111user_id_hi: ", user_id_hi)
            print ("111user_id_lo: ", user_id_lo)

            # trans hex string to dec string
            user_id_hi = int(str(user_id_hi), 16)
            user_id_lo = int(str(user_id_lo), 16)

            print ("222user_id_hi: ", user_id_hi)
            print ("222user_id_lo: ", user_id_lo)

            r = ClearOneUser(user_id_hi, user_id_lo)
            if r == ACK_SUCCESS:
                print ("Deleted successful !")
            elif r == ACK_FAIL:
                print ("Failed: Check again user id")
            else:
                print ("RESPONSE: r = ", r)

        else:
            print ("Wroing Command(must be 0 ~ 999): ", command)

    elif command == "5" and Finger_SleepFlag != 1:
        GPIO.output(Finger_RST_Pin, GPIO.LOW)
        Finger_SleepFlag = 1
        # print ("Module has entered sleep mode: you can use the finger Automatic wake-up function, in this mode, only CMD6 is valid, send CMD6 to pull up the RST pin of module, so that the module exits sleep !")
        print(" -- HOD (Hands on Detection) mode --")
    elif command == "6":
        Finger_SleepFlag = 0
        GPIO.output(Finger_RST_Pin, GPIO.HIGH)
        print ("The module is awake. All commands are valid !")
    elif command == "7" and Finger_SleepFlag != 1:
        print ("_________Request Image Data_________")
        print ("save fingerprint image (Put your finger on sensor until successfully/failed information returned) ")
        RequestImage()
        full_img = []
        print ("Wait a few seconds..")
        while 1:
            TxAndRxCmd(None, 96, 1)
            #print (len(g_rx_buf))
            for i in g_rx_buf:
                full_img.append(i)
            if g_rx_buf[-1] == 245:
                break

        if len(full_img) == 4611:
            full_img = full_img[1:-2]
            print ("(success) img data len : ", len(full_img))
        else:
            print ("Failed recv img data")


    else:
        pass
        # print ("commands are invalid !")

#***************************************************************************
# @brief   If you enter the sleep mode, then open the Automatic wake-up function of the finger,
#         begin to check if the finger is pressed, and then start the module and match
#***************************************************************************/
def Auto_Verify_Finger():
    while True:
        # If you enter the sleep mode, then open the Automatic wake-up function of the finger,
        # begin to check if the finger is pressed, and then start the module and match
        if Finger_SleepFlag == 1:
            if GPIO.input(Finger_WAKE_Pin) == 1:   # If you press your finger
                time.sleep(0.01)
                if GPIO.input(Finger_WAKE_Pin) == 1:
                    GPIO.output(Finger_RST_Pin, GPIO.HIGH)   # Pull up the RST to start the module and start matching the fingers
                    time.sleep(0.25)      # Wait for module to start
                    # print ("Waiting Finger......Please try to place the center of the fingerprint flat to sensor !")
                    print("Waiting Fingerprint....")
                    r = VerifyUser()
                    if r == ACK_SUCCESS:
                        print ("Matching successful: Door Open !")
                        Door_Open_Control("1")
                        print("________________________________________________________________________________")
                        print("_______________________Fingerprint Authentication Success_______________________")
                        print("________________________________________________________________________________")
                        raise Exception("PROCESS END")
                    elif r == ACK_NO_USER:
                        print ("Failed: Door close !")
                        Door_Open_Control("0")
                        # break
                    elif r == ACK_TIMEOUT:
                        pass
                        #print ("Failed: Time out !")
                    elif r == ACK_GO_OUT:
                        print ("Failed: Please try to place the center of the fingerprint flat to sensor !")

                    #After the matching action is completed, drag RST down to sleep
                    #and continue to wait for your fingers to press
                    GPIO.output(Finger_RST_Pin, GPIO.LOW)
        time.sleep(0.2)

#def Door_Open_Control(enable):
#    # stop motor
#    GPIO.output(12,GPIO.LOW)
#    GPIO.output(13,GPIO.LOW)
#    time.sleep(.2)
#
#    if enable == "1":
#        # door open
#        GPIO.output(13,GPIO.LOW)
#        GPIO.output(12,GPIO.HIGH)
#        time.sleep(3)
#
#    else:
#        # door close
#        GPIO.output(12,GPIO.LOW)
#        GPIO.output(13,GPIO.HIGH)
#        time.sleep(3)
#
#    # stop motor
#    GPIO.output(12,GPIO.LOW)
#    GPIO.output(13,GPIO.LOW)
#    time.sleep(.2)


def main():

    GPIO.output(Finger_RST_Pin, GPIO.LOW)
    time.sleep(0.25)
    GPIO.output(Finger_RST_Pin, GPIO.HIGH)
    time.sleep(0.25)    # Wait for module to start
    while SetCompareLevel(5) != 5:
        print ("***ERROR***: Please ensure that the module power supply is 3.3V or 5V, the serial line connection is correct.")
        time.sleep(1)
#    print ("***************************** WaveShare Capacitive Fingerprint Reader Test *****************************")
    print ("Compare Level:  5    (can be set to 0-9, the bigger, the stricter)")
    print ("Number of fingerprints already available:  %d "  % GetUserCount())
#    print (" send commands to operate the module: ")
#    print ("  CMD1 : Query the number of existing fingerprints")
#    print ("  CMD2 : Registered fingerprint  (Put your finger on the sensor until successfully/failed information returned) ")
#    print ("  CMD3 : Fingerprint matching  (Send the command, put your finger on sensor) ")
#    print ("  CMD4 : Clear fingerprints ")
#    print ("  CMD5 : Switch to sleep mode, you can use the finger Automatic wake-up function (In this state, only CMD6 is valid. When a finger is placed on the sensor,the module is awakened and the finger is matched, without sending commands to match each time. The CMD6 can be used to wake up) ")
#    print ("  CMD6 : Wake up and make all commands valid ")
#    print ("***************************** WaveShare Capacitive Fingerprint Reader Test ***************************** ")
    print ("  1 : 등록된 지문 개수 확인")
    print ("  2 : 지문 등록")
    print ("  3 : 지문 매칭")
    print ("  4 : 지문 지우기")
    print ("  5 : HOD(핸즈온디텍션)  모드")
    print ("  6 : wake up 모드")
    print ("  7 : 이미지 데이터 저장")

    # print(" 1 : Check the number of registered fingerprints")
    # print(" 2 : Register fingerprint")
    # print(" 3 : fingerprint matching")
    # print(" 4 : Erase fingerprint")
    # print(" 5 : HOD (Hands on Detection) mode")
    # print(" 6 : wake up mode")
    # print(" 7 : Save image data")


    t = threading.Thread(target=Auto_Verify_Finger)
    t.setDaemon(True)
    t.start()

    while  True:
        # option-1. manual command
        str = input("Input Command (1-6):")
        # option-2. fixed command (save image by jiseong)
        #str = "5"
        Analysis_PC_Command(str)

if __name__ == '__main__':
    try:
        main()
        # pid = os.getpid()
        # os.kill(pid, 2)

    except KeyboardInterrupt:
        if ser != None:
            ser.close()
        GPIO.cleanup()
        print("\n\n Test finished ! \n")
        sys.exit()


