clc
clear
close all
% measurenum=[30 24 18 12 6 29 23 17 11 5 28 22 16 10 4];


load ZigZagmove1mmps-3x6magnetHorizontal-Calibrated-Height10mmFromSensorBoard.mat

% for i=1:5406
%     for j=1:15
%         andazeh1(i,j)=sqrt(sen1to30(i,3*measurenum(j)-2)^2+sen1to30(i,3*measurenum(j)-1)^2+sen1to30(i,3*measurenum(j))^2);
% %     end
% % end
% for i=1:2:76866
%     for j=1:4:30
%         andazeh(i,j)=sqrt(sen1to30(i,3*(j)-2)^2+sen1to30(i,3*(j)-1)^2+sen1to30(i,3*(j))^2);
%     end
% end
comp=[];
for j=1:4:30
    comp=[comp sen1to30(1:2:76866,3*(j)-2) sen1to30(1:2:76866,3*(j)-1) sen1to30(1:2:76866,3*(j))];
end
comp1=comp(any(comp,2),:);
component=comp1(:,any(comp1,1));
X=(component)';
T=magnetpos(1:2:76866,1:2)';
net.output.processFcns = {'mapminmax'};
net = layrecnet(1,[20,10,5]);%[12 7 3]
net.trainFcn = 'trainscg';%trainscg trainrp trainoss
% net.layers{1}.transferFcn = 'radbas';
% net.layers{2}.transferFcn = 'radbas';
% net.layers{3}.transferFcn = 'radbas';
% net.outputs{1}.transferFcn = 'radbas';
% net.outputs{2}.transferFcn = 'radbas';
% net.trainParam.lr=0.1;%0.2
net.trainParam.max_fail=70;%20%60
% net.trainParam.mc=0.1;%0.05
net.trainParam.epochs=9500;%2000%9000
net = train(net,X,T,'useGPU','yes');
 
clear X T andazeh1 magnetpos andaze1 andazeh

load Curvemove1mmps-3x6magnetHorizontal-Calibrated-Height10mmFromSensorBoard.mat
comp=[];
for j=1:4:30
    comp=[comp sen1to30(1:9566,3*(j)-2) sen1to30(1:9566,3*(j)-1) sen1to30(1:9566,3*(j))];
end

comp1=comp(any(comp,2),:);
component=comp1(:,any(comp1,1));
% andazeh1(:,5)=zeros(9566,1);
%component = smooth(component,0.2,'rloess');
X=(component)';

T=magnetpos(1:9566,1:2)';

y =(net(X));
 smoothOutput1 = smooth(y(1,:),0.2,'rloess');
 smoothOutput2 = smooth(y(2,:),0.2,'rloess');


subplot(4,2,1)
plot(smoothOutput1)
hold on
plot(T(1,:))
legend('Estimated','Real')
xlabel('Samples')
ylabel('Position (X axis)')
title('Neural Network with Fault 8 sensors')

subplot(4,2,3)
plot(-smoothOutput2)
hold on
plot(T(2,:))

legend('Estimated','Real')
xlabel('Samples')
ylabel('Position (Y axis)')

subplot(4,2,5)
plot(smoothOutput1,-smoothOutput2)
hold on
plot(T(1,:),T(2,:))
legend('Estimated','Real')
xlabel('Position (X axis)')
ylabel('Position (Y axis)')

subplot(4,2,7)
plot(sum([(smoothOutput1-T(1,:)).^2;(-smoothOutput2-T(2,:)).^2]))
xlabel('Samples')
ylabel('Squered Error')
title(['Sumation of Squered Error is ',num2str(sum(sum([(smoothOutput1-T(1,:)).^2;(-smoothOutput2-T(2,:)).^2])))])


component(:,5)=zeros(9566,1);
%component = smooth
X=(component)';

T = magnetpos(:,1:2)';
y = net(X);
  smoothOutput1 = smooth(y(1,:),0.2,'rloess');
  smoothOutput2 = smooth(y(2,:),0.2,'rloess');

subplot(4,2,2)
plot(smoothOutput1)
hold on
plot(T(1,:))
legend('Estimated','Real')
xlabel('Samples')
ylabel('Position (X axis)')
title('Neural Network with Fault 8 sensors')

subplot(4,2,4)
plot(-smoothOutput2)
hold on
plot(T(2,:))

legend('Estimated','Real')
xlabel('Samples')
ylabel('Position (Y axis)')

subplot(4,2,6)
plot(smoothOutput1,-smoothOutput2)
hold on
plot(T(1,:),T(2,:))
legend('Estimated','Real')
xlabel('Position (X axis)')
ylabel('Position (Y axis)')

subplot(4,2,8)
plot(sum([(smoothOutput1-T(1,:)).^2;(-smoothOutput2-T(2,:)).^2]))
xlabel('Samples')
ylabel('Squered Error')
title(['Sumation of Squered Error is ',num2str(sum(sum([(smoothOutput1-T(1,:)).^2;(-smoothOutput2-T(2,:)).^2])))])

error1=sum(sum([(smoothOutput1-T(1,:)).^2;(-smoothOutput2-T(2,:)).^2]));
maxerror=max(max([abs(smoothOutput1-T(1,:));abs(-smoothOutput2-T(2,:))]))
