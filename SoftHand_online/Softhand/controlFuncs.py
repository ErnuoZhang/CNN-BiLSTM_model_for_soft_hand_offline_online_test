import struct
import serial

def checksum(data, length):
    chs = 0
    for i in range(length):
        A = chs & 0xFF
        B = data[i] & 0xFF
        chs = A ^ B
    return chs

def set_motor_inputs(s: serial.Serial, ID: int, val1: int, val2: int):
    CMD_SET_INPUTS = 130

    # Convert to little-endian uint8 representation of int16
    val1_bytes = struct.pack('<h', val1)  # 2 bytes, little endian
    val2_bytes = struct.pack('<h', val2)

    buf = [
        CMD_SET_INPUTS,
        val1_bytes[1],  # High byte
        val1_bytes[0],  # Low byte
        val2_bytes[1],  # High byte
        val2_bytes[0],  # Low byte
    ]

    chs = checksum(buf, 5)

    # Message format: ':' ':' ID 6 buf[0]...buf[4] chs
    message = bytearray()
    message.append(ord(':'))
    message.append(ord(':'))
    message.append(ID)
    message.append(6)
    message.extend(buf)
    message.append(chs)

    s.write(message)

def activate_device(s: serial.Serial, ID: int, activation: bool):
    CMD_ACTIVATE = 128
    ACTIVE = 3 if activation else 0

    buf = [CMD_ACTIVATE, ACTIVE]
    chs = checksum(buf, 2)

    # Construct message: ':' ':' ID 3 CMD_ACTIVATE ACTIVE chs
    message = bytearray()
    message.append(ord(':'))
    message.append(ord(':'))
    message.append(ID)
    message.append(3)
    message.extend(buf)
    message.append(chs)

    s.write(message)

def init_device(port: str) -> serial.Serial:
    s = serial.Serial(
        port=port,
        baudrate=2000000,
        bytesize=serial.EIGHTBITS,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.01  # 10 ms timeout
    )
    return s

