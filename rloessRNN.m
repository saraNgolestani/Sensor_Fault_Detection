 

smoothOutput1 = smoothdata(y(1,:),'rloess',[5 0]);
smoothOutput2 = smoothdata(y(2,:),'rloess',[5 0]);

subplot(4,1,1)
plot(smoothOutput1)
hold on
plot(T(1,:))
legend('NN1','Real','NN2')
xlabel('Samples')
ylabel('Position (X axis)')
title('Neural Network with Fault 8 sensors')

subplot(4,1,2)
plot(-smoothOutput2)
hold on
plot(T(2,:))

legend('NN1','Real','NN2')
xlabel('Samples')
ylabel('Position (Y axis)')

subplot(4,1,3)
plot(smoothOutput1,-smoothOutput2)
hold on
plot(T(1,:),T(2,:))
legend('NN1','Real','NN2')
xlabel('Position (X axis)')
ylabel('Position (Y axis)')

subplot(4,1,4)
plot(sum([(smoothOutput1-T(1,:));(-smoothOutput2-T(2,:))]))
legend('NN1','NN2','NN2')
xlabel('Samples')
ylabel('Squered Error')
%title(['Sumation of Squered Error is ',num2str(sum(sum([(smoothOutput1-T(1,:)).^2;(-smoothOutput2-T(2,:)).^2])))])

error1=sum(sum([(smoothOutput1-T(1,:)).^2;(-smoothOutput2-T(2,:)).^2]));
maxerror1=max(max([abs(smoothOutput1-T(1,:));abs(-smoothOutput2-T(2,:))]))
error1=sum(sum([(y(1,:)-T(1,:)).^2;(-y(2,:)-T(2,:)).^2]));
maxerror=max(max([abs(y(1,:)-T(1,:));abs(-y(2,:)-T(2,:))]))
MaxErrAmp=max(sqrt(sum([(smoothOutput1-T(1,:)).^2;(-smoothOutput2-T(2,:)).^2])))

