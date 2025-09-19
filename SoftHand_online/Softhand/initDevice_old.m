function [s] = initDevice_old(port)

%     s = serial(port,2000000);
    s = serial(port,'BaudRate',2000000,'DataBits',8);
    set(s,'StopBits',1);
%     set(s,'DataBits',8);
    set(s,'Timeout',0.01);  % 10 msec. timeout on read
 
end

