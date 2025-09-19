
%open device
s = initDevice_old("COM5");
fopen(s);
ID = 1;
activateDevice(s,ID,true);


%send comand
setMotorInputs(s,ID,0,0);

pause(3);

setMotorInputs(s,ID,0,19000);


pause(3);

setMotorInputs(s,ID,0,0);

pause(3);

setMotorInputs(s,ID,-13000,0);

pause(3);

setMotorInputs(s,ID,0,0);

pause(3);

setMotorInputs(s,ID,-13000,19000);

pause(3);

setMotorInputs(s,ID,0,0);




% close device
setMotorInputs(s,ID,0,0);

pause(3);

activateDevice(s,ID,false);
fclose(instrfind);
delete(s);



