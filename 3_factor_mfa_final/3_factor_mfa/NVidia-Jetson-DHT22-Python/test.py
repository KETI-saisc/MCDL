import time
import C_DHT

while True:
    time.sleep(1)
    C_DHT.readSensor(0)        # if used with DHT22, read first sensor
    C_DHT.readSensorDHT11(0)   # if used with DHT11 sensors
