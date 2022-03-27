function [averagePerRAO, avPreamStatsPerRAO,Qnet,vectorPacb, Ps, PsM, PsH, K, EK, KM, EKM, KH, EKH, D, ED, D95, DM, EDM, D95M, DH, EDH, D95H, counterexpout, memoryexpout] = LTEA_M_H_ACB_DDQL_v2(arrivalsM, arrivalsH,Qnet,epsilon,gamma,alpha,Pacbinicio,counterexp,memoryexp,tamexp,updatetarget)
%-------------------------------------------------
%ultima actualizacion Julio 22 2018: se revisa que el buffer sea continuo
%con cada iteracion
% La idea es implementar 1) target network
%                        2) experience replay
%                        3) double deep q-learning: En la versión de
%                        van Hasselt
% Usando las mismas rewards de lo que se resolvio con Double q-learning y
% q-learning vanilla.
% Algunas referencias:
% 1) Deep Reinforcement Learning with Double Q-learning 
%  Hado van Hasselt and Arthur Guez and David Silver
% 2) https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/Double%20DQN%20Solution.ipynb
% 3) https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
% En 1 proponen usar la red de target network para double dqn.



%-------------------------------------------------
% [averagePerRAO, Ps, PsM, PsH, K, EK, KM, EKM, KH, EKH, D, ED, D95, DM, EDM, D95M, DH, EDH, D95H] = LTEA_M_H_ACB(arrivalsM, arrivalsH)
% [averagePerRAO, avPreamStatsPerRAO, Ps, PsM, PsH, K, KH, KM, D, DH, DM, successfulUEs, successfulUEsM, successfulUEsH]
% Use [a b c d] = LTEA_M_H(x, y) to call the function;
%====== Arrivals Matrix ========
% beta.a = 3; beta.b = 4;totalUEs = 30000; numSimulations = 10
% arrivalsM = 1e4.*betarnd(beta.a,beta.b,totalUEs,numSimulations); %distribution
% of M2M UEs, 1e4 [ms]
% maxTime = 10*60*1000; connections = 33000 (7937*4.1581), numSimulations = 10
% arrivalsH = unifrnd(0,maxTime,connections,numSimulations); % distribution of H2H UEs
% Parameters Configuration
%======= Channel Parameters ===========
%Available preambles 54
%RAO periodicity 5
%Number of uplink grants per subframe 3
%Maximum number of preamble transmission attempts 10
%Backoff indicator 20
%======= Delay parameters ==========
%Contention resolution timer 48
%Preamble processing delay 2
%RAR processing delay 5
%Connection request processing delay 4
%Number of Msg3 and Msg4 transmission attempts 5
%HARQ re-transmission probability for Msg3 and Msg4  0.1
%RTT for Msg3 8
%RTT for Msg4 5
% ===== ACB parameters ========
acb.prob = 0.5; % Probability
acb.time = 4e3; % Time [ms]
%====================================
% maxRAOs = (10*60*1000)/5; %[ms] --> #RAOs in 10 minutes
maxRAOs = 2e4;
%Prueba junio 24 2019
RACHConfig = rach(54,5,3,10,20,48,2,5,4,5,0.1,8,5);
%RACHConfig = rach(54,1,3,10,20,48,2,5,4,5,0.1,8,5); %en cada ms hay un RAO
typeM = 1;
typeH = 2;
% -----------------------------------
K = zeros(RACHConfig.maxNumPreambleTxAttempts,1); % Vector with the distribution
% of preamble transmissions
KM = zeros(RACHConfig.maxNumPreambleTxAttempts,1);
KH = zeros(RACHConfig.maxNumPreambleTxAttempts,1);
D = zeros(maxRAOs*5,1); % Vector with the distribution of access delay [ms]
DM = zeros(maxRAOs*5,1);
DH = zeros(maxRAOs*5,1);
ueArrivals = [arrivalsM;arrivalsH];
[totalUEsM, ~] = size(arrivalsM);
[totalUEsH, ~] = size(arrivalsH);
[totalUEs, numSimulations] = size(ueArrivals);
preambleDetectionProbability = 1-1./(exp(1:RACHConfig.maxNumPreambleTxAttempts));
totalSuccessfulUEs = 0;
totalSuccessfulUEsM = 0;
totalSuccessfulUEsH = 0;
totalFailedUEsH = 0;
statsPerRAO = zeros(maxRAOs,5,numSimulations); % Matrix with the:
% [1 FirstPreamTx, 2 Total Access attempts, 3 Collisions, 4 Successfully
% decoded preambles, 5 successful accesses per RAO]
preambleStatsPerRAO = zeros(maxRAOs,3,numSimulations); %Matrix with the:
% [1 Successful preambles 2 Not used preambles 3 Collided preambles]
successfulUEs =  zeros(numSimulations,1);
successfulUEsM = zeros(numSimulations,1);
successfulUEsH = zeros(numSimulations,1);
failedUEsM = zeros(numSimulations,1);
failedUEsH = zeros(numSimulations,1);

%====================================
% Definition of parameters for Q-learning

PreambleTransM=[0:1:29]; %Este valor es maximo 29, valores mas grandes a 29 los aproximo a 29
Pacb=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1];
PreambleTransCV=[0:0.2:1];% Todos los valores se aproximan a 0, 0.2, 0.4 0.6 y 0.8
DeltaNpsref=[1:1:3];%1 crecio,2 disminuyo, 3 igual : 3 estados

Nstates=length(PreambleTransM)*length(Pacb)*length(PreambleTransCV)*length(DeltaNpsref);
%The set of actions A, allow to change Pacb.

action= [1:1:16];

% secondary  Q matrix
%SQ = zeros(Nstates,length(action)); 
%capas=10;
%SQnet=feedforwardnet(capas);
%SQnet = train(SQnet,[0;0;0;0],[0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0]);
SQnet=Qnet;

%Q=zeros(30*16*3*5,16);
%We set the values for q-learning
%gamma=0.7;
%alpha=0.2;
%epsilon=0.01;% This value will change as time passes by and after a while it should be 0.

%Period of SIB measured in RAOs: 80 ms
TSIB=16; %cambio en junio 24 de 2019

%currentstate=convertvarstorow(0,1);

actionref=16; %this is the action made on the last decision period

windowCC=TSIB;   %This is the size of the window to measure the CC in RAOs
Transpreamblehistory=zeros(windowCC,1);
Pacbref=1;
Pacbcur=1;
Npsref=0;
Npscur=0;
NpsMref=0;
NpsMcur=0;
NpsCVref=0;
NpsCVcur=0;
DeltaNpsref=0;
DeltaNpscur=0;
actioncur=16;
actionref=16;
numSuccessfullyDecodedPreambles=0;
vectorPacb=ones(maxRAOs,1);

%matriz para entrenamiento
%s,a,s',r
%tamexp=500;
%cambio julio 13
%tamexp=800;

%counterexp=1;
%memoryexp=zeros(tamexp,26);

%numero de actualizacion target
%updatetarget=100;
%cambio de este valor para evaluarlo JUlio 13 de 2019
%updatetarget=400;

%====================================


for simulation = 1:numSimulations
    %     successfulUEs = 0;
    %     successfulUEsM = 0;
    %     successfulUEsH = 0;
    
    %     failedUEsM = 0;
    arrivalTime = ueArrivals(:,simulation);
    arrivalTime = reshape(arrivalTime,totalUEs,1);
    UEs = [zeros(totalUEs,1),arrivalTime,zeros(totalUEs,4)]; % UEs matrix with:
    %[1 ArrivalTime, 2 AccessTime, 3 #PreambleTransmissions, 4 SelectedPreamble
    % 5 AC 6 Type]
    UEs(1:totalUEsM,6) = typeM; % for identifying M2M UEs
    UEs(totalUEsM+1:totalUEs,6) = typeH; % for identifying H2H UEs
    
    simulationTime = 0;
    RAO = 0;
    while(successfulUEsM(simulation)+failedUEsM(simulation)<totalUEsM && RAO<maxRAOs) % For each RAO do:
        RAO = RAO+1;
        simulationTime = simulationTime + RACHConfig.raoPeriodicity;
        accessesInRAO = find(UEs(:,2)<=simulationTime); % Find the accessing UEs
        
        %=================================================================
        % This section is for DDQL: Part 1
        
         
        
        if floor((RAO-1)/TSIB)==((RAO-1)/TSIB) %It is a RAO with SIB
            %A decision must be made about the PACB for this cycle
            
            actionvar=rand;
            
            if actionvar<=epsilon %then the next action is random
                    
                actioncur=randi([1 16],1,1);        
            else %then the next action is the one that maximizes Q

                %Aca hay que reemplazar para que la accion la encuentre con
                %la red neuronal de Q
                %memoryexp(counterexp,1)=NpsMref;
                %memoryexp(counterexp,2)=NpsCVref;
                %memoryexp(counterexp,3)=DeltaNpsref;
                %memoryexp(counterexp,4)=Pacbref;
                [varinutil actioncur]=max(Qnet([NpsMcur;NpsCVcur;DeltaNpscur;Pacbcur]));
                
                
                %[varinutil actioncur]=max(Q(convertvarstostateQ8(NpsMcur,NpsCVcur,DeltaNpscur,Pacbcur),:));
  
            end
            
            %Now we update Pacb based on actioncur
          %Pacbref=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1] %16 estados
                actionref=actioncur;
                Pacbref=Pacbcur; %se actualiza porque hace parte de la accion
            
                if actioncur==1
                    Pacbcur=0.05;
                elseif actioncur==2 
                    Pacbcur=0.1;
                elseif actioncur==3 
                    Pacbcur=0.15;
                elseif actioncur==4 
                    Pacbcur=0.2;
                elseif actioncur==5 
                    Pacbcur=0.25;
                elseif actioncur==6 
                    Pacbcur=0.3;
                elseif actioncur==7 
                    Pacbcur=0.4;
                elseif actioncur==8 
                    Pacbcur=0.5;
                elseif actioncur==9 
                    Pacbcur=0.6;
                elseif actioncur==10 
                    Pacbcur=0.7;
                elseif actioncur==11 
                    Pacbcur=0.75;
                elseif actioncur==12 
                    Pacbcur=0.8;
                elseif actioncur==13 
                    Pacbcur=0.85;
                elseif actioncur==14 
                    Pacbcur=0.9;
                elseif actioncur==15 
                    Pacbcur=0.95;
                else
                    Pacbcur=1;
                end
                                 
        else %It is a RAO w/o SIB
            % PACB does not change           
        end

        acb.prob=Pacbcur;
        vectorPacb(RAO)=acb.prob;
        %=================================================================
                
        
        if(accessesInRAO)
            newArrivals = find(UEs(accessesInRAO,1)==0); % Find the new arrivals
            %             statsPerRAO(RAO,1,simulation) = statsPerRAO(RAO,1,simulation) + ...
            %                 length(newArrivals);
            UEs(accessesInRAO(newArrivals),1) = simulationTime;
            %             % ============= Access Class Barring ===============
            withoutACB = find(UEs(accessesInRAO,3)==0); % Find the UEs subjecto
            % to the ACB scheme
            if(withoutACB)
                UEs(accessesInRAO(withoutACB),2) = simulationTime +...
                    (rand(size(withoutACB))>acb.prob).*...
                    (0.7+0.6.*rand(size(withoutACB))).*acb.time;
            end
            %             statsPerRAO(RAO,6,simulation) = statsPerRAO(RAO,1,simulation) + ...
            %                 length(withoutACB); %arrivals after ACB
            %             % ---------------------------------------------------
            accessesInRAO = find(UEs(:,2)<=simulationTime); % Find the accessing UEs
            firstPreamTx = (UEs(:,2)<=simulationTime & UEs(:,3)==0);
            statsPerRAO(RAO,1,simulation) = statsPerRAO(RAO,1,simulation) + ...
                nnz(firstPreamTx);
            statsPerRAO(RAO,2,simulation) = statsPerRAO(RAO,2,simulation) + ...
                length(accessesInRAO);
            UEs(accessesInRAO,4) = unidrnd(RACHConfig.availablePreambles, ...
                size(accessesInRAO)); % The UEs select a preamble randomly
            UEs(accessesInRAO,3) = UEs(accessesInRAO,3)+1;  % The UEs send the
            % preamble and increase the preamble transmission counter
            selectedPreambles = zeros(size(accessesInRAO));
            for i = 1:length(accessesInRAO)   % Identify the preambles transmitted by only one UE
                selectedPreambles(i) = (sum(UEs(accessesInRAO,4)==UEs(accessesInRAO(i),4)));
            end
            successfulPreambles = (selectedPreambles==1);
            preambleStatsPerRAO(RAO,1,simulation) = sum(successfulPreambles);
            preambleStatsPerRAO(RAO,2,simulation) = RACHConfig.availablePreambles - ...
                length(unique(UEs(accessesInRAO,4)));
            preambleStatsPerRAO(RAO,3,simulation) = RACHConfig.availablePreambles - ...
                preambleStatsPerRAO(RAO,1,simulation) - preambleStatsPerRAO(RAO,2,simulation);
            statsPerRAO(RAO,3,simulation) = statsPerRAO(RAO,3,simulation) + sum(1-successfulPreambles);
            successfullyDecodedPreambles = successfulPreambles.*(rand(size(successfulPreambles))<preambleDetectionProbability(UEs(accessesInRAO,3))');	%Decode the preambles
            numSuccessfullyDecodedPreambles = sum(successfullyDecodedPreambles);
            statsPerRAO(RAO,4,simulation) = statsPerRAO(RAO,4,simulation) + numSuccessfullyDecodedPreambles;
            waitingUG = accessesInRAO(successfullyDecodedPreambles==1); % Find UEs that sent a correctly decoded preamble
            j = unidrnd(numSuccessfullyDecodedPreambles)-1;
            %             failedUEsM = 0;
            for i = 1:min(numSuccessfullyDecodedPreambles,RACHConfig.raoPeriodicity*RACHConfig.uplinkGrantsPerSubframe)
                delayUG = floor((i-1)/RACHConfig.uplinkGrantsPerSubframe)+1; % Assign uplink grants
                HARQmsg3 = 0;
                HARQmsg4 = 0;
                Tmsg4 = 0;
                while(rand()<RACHConfig.harqReTxProbMsg3Msg4 && HARQmsg3<RACHConfig.maxNumMsg3Msg4TxAttempts)	% Msg 3 transmission
                    HARQmsg3 = HARQmsg3+1;
                end
                while(rand()<RACHConfig.harqReTxProbMsg3Msg4 && HARQmsg3<RACHConfig.maxNumMsg3Msg4TxAttempts ...
                        && HARQmsg4<RACHConfig.maxNumMsg3Msg4TxAttempts) % Msg4 transmission
                    HARQmsg4 = HARQmsg4+1;
                end
                contentionResolutionDelay = HARQmsg3*RACHConfig.rttMsg3 +...
                    HARQmsg4*RACHConfig.rttMsg4 + RACHConfig.connectionRequestProcessingDelay...
                    + 2 + Tmsg4;	% Obtain the delay of transmitting Msg3 and Msg4
                if(HARQmsg3<RACHConfig.maxNumMsg3Msg4TxAttempts && HARQmsg4<RACHConfig.maxNumMsg3Msg4TxAttempts...
                        && contentionResolutionDelay<=RACHConfig.contentionResolutionTimer)	% Sucessful access
                    delay = simulationTime + 1 + RACHConfig.preambleProcessingDelay...
                        + delayUG + RACHConfig.rarProcessingDelay + contentionResolutionDelay...
                        - UEs(waitingUG(j+1),1);	% Calculate the overall delay [ms]
                    D(delay) = D(delay)+1;
                    UEs(waitingUG(j+1),2) = 0;
                    successfulUEs(simulation) = successfulUEs(simulation)+1;
                    if(UEs(waitingUG(j+1),6) == typeM)
                        successfulUEsM(simulation) = successfulUEsM(simulation)+1;
                        DM(delay) = DM(delay)+1;
                        KM(UEs(waitingUG(j+1),3)) = KM(UEs(waitingUG(j+1),3))+1;
                    elseif(UEs(waitingUG(j+1),6) == typeH)
                        successfulUEsH(simulation) = successfulUEsH(simulation)+1;
                        DH(delay) = DH(delay)+1;
                        KH(UEs(waitingUG(j+1),3)) = KH(UEs(waitingUG(j+1),3))+1;
                    end
                    statsPerRAO(RAO,5,simulation) = statsPerRAO(RAO,5,simulation)+1;
                    K(UEs(waitingUG(j+1),3)) = K(UEs(waitingUG(j+1),3))+1;
                else
                    UEs(waitingUG(j+1),2) = simulationTime + RACHConfig.backoffIndicator*rand()...
                        + 1 + RACHConfig.preambleProcessingDelay + delayUG + RACHConfig.rarProcessingDelay...
                        + contentionResolutionDelay; % Contention resolution failed, backoff
                end
                j = j+1;
                j = mod(j,numSuccessfullyDecodedPreambles);
            end
            successfulUEAccesses = UEs(:,2)==0; % Find the successful UEs
            %             successfulUEAccessesH = (UEs(:,2)==0 && UEs(:,2)==1);
            %             successfulUEAccessesM = (UEs(:,2)==0 && UEs(:,2)==2);
            UEs(successfulUEAccesses,:) = []; % Erase the successful UEs from the matrix of UEs
            terminateRAP = UEs(:,3) == RACHConfig.maxNumPreambleTxAttempts; % Find the UEs that failed in their last preamble transmission
            terminateRAPM = (UEs(:,3) == RACHConfig.maxNumPreambleTxAttempts)...
                & (UEs(:,6) == typeM);
            terminateRAPH = (UEs(:,3) == RACHConfig.maxNumPreambleTxAttempts)...
                & (UEs(:,6) == typeH);
            failedUEsM(simulation) = failedUEsM(simulation) + nnz(terminateRAPM);
            failedUEsH(simulation) = failedUEsH(simulation) + nnz(terminateRAPH);
            UEs(terminateRAP,:) = []; % Erase the UEs that failed in all their preamble transmissions
            backoffUEs = UEs(:,2)<=simulationTime;
            if(nnz(backoffUEs)>0) % The UEs that failed and that still can perform preamble transmissions perform backof
                UEs(backoffUEs,2) = simulationTime + 1 + RACHConfig.preambleProcessingDelay...
                    + RACHConfig.raoPeriodicity + RACHConfig.backoffIndicator*rand(nnz(backoffUEs),1);
            end
        end
        
        %=================================================================
        % This section is for Q-learning: Part 2
        
        %First we save the info of preambles sent
        
        if RAO<=(windowCC)
            Transpreamblehistory(RAO)=numSuccessfullyDecodedPreambles;
        else
            Transpreamblehistory=circshift(Transpreamblehistory,1);
            Transpreamblehistory(windowCC)=numSuccessfullyDecodedPreambles;
        end
        
        %Now we recover the new values of Nrs
        Npscur=numSuccessfullyDecodedPreambles;
         
                %fileID = fopen('experimento.txt','a');
                %fprintf(fileID,'Pacb: %f \n',Pacbcur);
                %fprintf(fileID,'Npscur: %d \n',Npscur);
                %fclose(fileID);
         
         
         if floor((RAO)/TSIB)==((RAO)/TSIB) % entra en el rao antes de entrar al SIB y despues de los accesos
             

                    %Actualiza el estado anterior y el estado actual.
                    %Junto con el reward debe ser guardado en memoria.
                    %-----------------------------------------------
             
                    NpsMref=NpsMcur;
                    NpsCVref=NpsCVcur;
                    DeltaNpsref=DeltaNpscur;
                    
                    
                    
                    NpsMcur=mean(Transpreamblehistory);
                    
                    if NpsMcur>0
                        NpsCVcur=(var(Transpreamblehistory)^0.5/mean(Transpreamblehistory));
                    else
                        NpsCVcur=0;
                    end
                    
                    NpsMcur=round(NpsMcur);
                    
                    if (NpsCVcur>=0) && (NpsCVcur<0.2)
                        NpsCVcur=0;
                    elseif (NpsCVcur>=0.2) && (NpsCVcur<0.4)
                        NpsCVcur=0.2;
                    elseif (NpsCVcur>=0.4) && (NpsCVcur<0.6)
                        NpsCVcur=0.4;
                    elseif (NpsCVcur>=0.6) && (NpsCVcur<0.8)
                        NpsCVcur=0.6;   
                    elseif (NpsCVcur>=0.8)
                        NpsCVcur=1;       
                    end                    
                    
                    DifNps=NpsMcur-NpsMref;
                    
                    if DifNps>0
                        DeltaNpscur=1;
                    elseif DifNps==0
                        DeltaNpscur=3;
                    else 
                        DeltaNpscur=2;
                    end
                    
                    %DeltaNpsref=DeltaNpscur;
                    
         
            %Now we can calculate the reward of the action taken on Part 1
            %for the last SIB time.
            %           
               %___________________
                
                reward = calculatereward4(NpsMcur,NpsCVcur,DeltaNpscur,Pacbcur);    
               %___________________
                 
%                 fileID = fopen('experimento.txt','a');
%                 fprintf(fileID,'----------------RAO CON DECISION ------------------\n');
%                 fprintf(fileID,'RAO: %d \n',RAO);
%                 fprintf(fileID,'NpsMref: %f \n',NpsMref);
%                 fprintf(fileID,'NpsMcur: %f \n',NpsMcur);
%                 fprintf(fileID,'NpsCVref: %f \n',NpsCVref);
%                 fprintf(fileID,'NpsCVcur: %f \n',NpsCVcur);
%                 fprintf(fileID,'DeltaNpsref: %f \n',DeltaNpsref);
%                 fprintf(fileID,'DeltaNpscur: %f \n', DeltaNpscur);
%                 fprintf(fileID,'Pacbref: %f \n',Pacbref);
%                 fprintf(fileID,'Pacbcur: %f \n',Pacbcur);
%                 fprintf(fileID,'reward: %f \n', reward);                
%                 fprintf(fileID,'Preambulos: %f \n',preambleStatsPerRAO(RAO,1,simulation));
%                 fprintf(fileID,'Preambulos libres: %f \n',preambleStatsPerRAO(RAO,2,simulation));
%                 fprintf(fileID,'Preambulos con colision: %f \n',preambleStatsPerRAO(RAO,3,simulation));
%                 fprintf(fileID,'Qoriginal: %f \n', Q(convertvarstostateQ8(NpsMref,NpsCVref,DeltaNpsref,Pacbref),actioncur));
% 
%                 fclose(fileID);   
                
                
                % Ahora se guarda la información en la matriz de
                % experiencia
                %---------------------------------------------------------
                memoryexp(counterexp,1)=NpsMref;
                memoryexp(counterexp,2)=NpsCVref;
                memoryexp(counterexp,3)=DeltaNpsref;
                memoryexp(counterexp,4)=Pacbref;
                memoryexp(counterexp,5)=actioncur;
                memoryexp(counterexp,6)=NpsMcur;
                memoryexp(counterexp,7)=NpsCVcur;
                memoryexp(counterexp,8)=DeltaNpscur;
                memoryexp(counterexp,9)=Pacbcur;
                memoryexp(counterexp,10)=reward;
                
              
                %1)Encontrar la accion a' que maximiza la matriz online en s' (Qnet)
                
%                 [varinutil actionmaxsprima]=max(Qnet([NpsMcur;NpsCVcur;DeltaNpscur;Pacbcur]));
                
                %2)Evaluar la accion a' en la matriz de ref SQnet en estado
                %s'
                
%                 Qevaluado=SQnet([NpsMcur;NpsCVcur;DeltaNpscur;Pacbcur]);
                %valorQevaluado=Qevaluado(actionmaxsprima);
                
                %3) Calcular el target = reward + gamma Q(s',a')
                
                %target=reward+gamma*valorQevaluado;
%                 target=reward+gamma*Qevaluado(actionmaxsprima);
                
                %4) entrenar para que Qnet(s,a)=target
                
                %net = train(net,x,t);
                
                %hay que generar el vector de salida deseado tomando
                %target
%                 vectorsalida=Qnet([NpsMref;NpsCVref;DeltaNpsref;Pacbref]);
%                 vectorsalida(actionmaxsprima)=target
                   
                %se llenan el resto de posiciones de memoria para su
                %posterior entrenamiento
                
%                 memoryexp(counterexp,11)=vectorsalida(1);
%                 memoryexp(counterexp,12)=vectorsalida(2);
%                 memoryexp(counterexp,13)=vectorsalida(3);
%                 memoryexp(counterexp,14)=vectorsalida(4);
%                 memoryexp(counterexp,15)=vectorsalida(5);
%                 memoryexp(counterexp,16)=vectorsalida(6);
%                 memoryexp(counterexp,17)=vectorsalida(7);
%                 memoryexp(counterexp,18)=vectorsalida(8);
%                 memoryexp(counterexp,19)=vectorsalida(9);
%                 memoryexp(counterexp,20)=vectorsalida(10);
%                 memoryexp(counterexp,21)=vectorsalida(11);
%                 memoryexp(counterexp,22)=vectorsalida(12);
%                 memoryexp(counterexp,23)=vectorsalida(13);
%                 memoryexp(counterexp,24)=vectorsalida(14);
%                 memoryexp(counterexp,25)=vectorsalida(15);
%                 memoryexp(counterexp,26)=vectorsalida(16);
                
                %aumenta la posicion de la memoria
                
                counterexp=counterexp+1;
                
                %matriz para entrenamiento
                %s,a,s',r
                %tamexp=5000;
                %counterexp=1;
                %memoryexp=zeros(tamexp,10);
                %PreambleTransM=[0:1:29]; %Este valor es maximo 29, valores mas grandes a 29 los aproximo a 29
                %Pacb=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1];
                %PreambleTransCV=[0:0.2:1];% Todos los valores se aproximan a 0, 0.2, 0.4 0.6 y 0.8
                %DeltaNpsref=[1:1:3];%1 crecio,2 disminuyo, 3 igual : 3 estados

                
                
                
                % We calculate Q
                
                %Qrownextaction=max(Q(convertvarstostateQ8(NpsMcur,NpsCVcur,DeltaNpscur,Pacbcur),:));
            
                %Q(convertvarstostateQ8(NpsMref,NpsCVref,DeltaNpsref,Pacbref),actioncur)= Q(convertvarstostateQ8(NpsMref,NpsCVref,DeltaNpsref,Pacbref),actioncur) + alpha*(reward+(gamma*Qrownextaction)-Q(convertvarstostateQ8(NpsMref,NpsCVref,DeltaNpsref,Pacbref),actioncur));

%                 fileID = fopen('experimento.txt','a');
%                 fprintf(fileID,'Qmodificado: %f \n', Q(convertvarstostateQ8(NpsMref,NpsCVref,DeltaNpsref,Pacbref),actioncur));
%                 fprintf(fileID,'---------------------------------------------\n');
%                 fclose(fileID); 
         
         else
             %Aca entra cuando no se toma accion, solo llegan preambulos.
             
%                 fileID = fopen('experimento.txt','a');
%                 fprintf(fileID,'RAO: %d \n',RAO);
%                 fprintf(fileID,'Pacbref: %f \n',Pacbref);
%                 fprintf(fileID,'Pacbcur: %f \n',Pacbcur);
%                 fprintf(fileID,'NpsMcur: %f \n',NpsMcur);
%                 fprintf(fileID,'Preambulos: %f \n',preambleStatsPerRAO(RAO,1,simulation));
%                 fprintf(fileID,'Preambulos libres: %f \n',preambleStatsPerRAO(RAO,2,simulation));
%                 fprintf(fileID,'Preambulos con colision: %f \n',preambleStatsPerRAO(RAO,3,simulation));
%                   fprintf(fileID,'---------------------------------------------\n');
%                 fclose(fileID);
         end
         
         %%Cuando guarda suficiente información en la memoria debe
         %%entrenarse
         
         if counterexp-1==tamexp
            counterexp=1; % lo resetea
            
                
                %genera una nueva matriz con las experiencias ordenadas de
                %manera aleatoria
                
                random_mem = memoryexp(randperm(size(memoryexp, 1)), :);
               
                
                %se entrena la NN con updateget datos
                %esto se hace tantas veces como updatetarget quepa en la
                %memoria
                
                for nentrena=1:(tamexp/updatetarget)
                    
                    %calcula los valores de target para cada uno de los
                    %valores de memoria con los cuales va a entrenar
                    
                    for calculatarget=((nentrena-1)*updatetarget)+1:nentrena*updatetarget
                    
                         %1)Encontrar la accion a' que maximiza la matriz online en s' (Qnet)
                
                        [varinutil actionmaxsprima]=max(Qnet([random_mem(calculatarget,6);random_mem(calculatarget,7);random_mem(calculatarget,8);random_mem(calculatarget,9)]));
                
                        %2)Evaluar la accion a' en la matriz de ref SQnet en estado
                        %s'
                
                        Qevaluado=SQnet([random_mem(calculatarget,6);random_mem(calculatarget,7);random_mem(calculatarget,8);random_mem(calculatarget,9)]);
                        %valorQevaluado=Qevaluado(actionmaxsprima);
                
                        %3) Calcular el target = reward + gamma Q(s',a')
                
                        %target=reward+gamma*valorQevaluado;
                        target=random_mem(calculatarget,10)+gamma*Qevaluado(actionmaxsprima);
                
                        %4) entrenar para que Qnet(s,a)=target
                
                        %net = train(net,x,t);
                
                        %hay que generar el vector de salida deseado tomando
                        %target
                        vectorsalida=Qnet([random_mem(calculatarget,1);random_mem(calculatarget,2);random_mem(calculatarget,3);random_mem(calculatarget,4)]);
                        %se reemplaza la posicion de a, es decir de Q(s,a),
                        %la accion original, es decir actioncur
                        vectorsalida(random_mem(calculatarget,5))=target;
                        
                        random_mem(calculatarget,11)=vectorsalida(1);
                        random_mem(calculatarget,12)=vectorsalida(2);
                        random_mem(calculatarget,13)=vectorsalida(3);
                        random_mem(calculatarget,14)=vectorsalida(4);
                        random_mem(calculatarget,15)=vectorsalida(5);
                        random_mem(calculatarget,16)=vectorsalida(6);
                        random_mem(calculatarget,17)=vectorsalida(7);
                        random_mem(calculatarget,18)=vectorsalida(8);
                        random_mem(calculatarget,19)=vectorsalida(9);
                        random_mem(calculatarget,20)=vectorsalida(10);
                        random_mem(calculatarget,21)=vectorsalida(11);
                        random_mem(calculatarget,22)=vectorsalida(12);
                        random_mem(calculatarget,23)=vectorsalida(13);
                        random_mem(calculatarget,24)=vectorsalida(14);
                        random_mem(calculatarget,25)=vectorsalida(15);
                        random_mem(calculatarget,26)=vectorsalida(16);
                    
                    end
                    
                    %entrenar la red con updatetarger muestras
                    
                    Qnet = train(Qnet, [random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,1)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,2)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,3)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,4)'],[random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,11)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,12)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,13)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,14)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,15)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,16)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,17)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,18)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,19)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,20)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,21)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,22)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,23)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,24)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,25)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,26)']);
                    
                    %No se que hace esta linea
                    random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,1);
                    
                    %view(Qnet)
                    
                    %y=Qnet([random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,1)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,2)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,3)';random_mem(((nentrena-1)*updatetarget)+1:nentrena*updatetarget,4)']);
                 
                    SQnet=Qnet;
                    
                end
                
                 %Actualiza SQnet
                 %SQnet=Qnet;
                
                %Qnet= train(Qnet,[memoryexp(ordenleer(memoria),1);memoryexp(ordenleer(memoria),2);memoryexp(ordenleer(memoria),3);memoryexp(ordenleer(memoria),4)],[memoryexp(ordenleer(memoria),11);memoryexp(ordenleer(memoria),12);memoryexp(ordenleer(memoria),13);memoryexp(ordenleer(memoria),14);memoryexp(ordenleer(memoria),15);memoryexp(ordenleer(memoria),16);memoryexp(ordenleer(memoria),17);memoryexp(ordenleer(memoria),18);memoryexp(ordenleer(memoria),19);memoryexp(ordenleer(memoria),20);memoryexp(ordenleer(memoria),21);memoryexp(ordenleer(memoria),22);memoryexp(ordenleer(memoria),23);memoryexp(ordenleer(memoria),24);memoryexp(ordenleer(memoria),25);memoryexp(ordenleer(memoria),26);]);
                
                %probar a ver que tal
                %y=Qnet([memoryexp(ordenleer(memoria),1);memoryexp(ordenleer(memoria),2);memoryexp(ordenleer(memoria),3);memoryexp(ordenleer(memoria),4)])
                
                %SQnet=Qnet
                
                %entrena de nuevo con el resto de datos
                
                
                
%                if mod(memoria,updatetarget)==0  %actualiza SQnet
                   
                    %hacer que SQnet=Qnet
%                    SQnet=Qnet;
%                else
%                end
                                   
                
            %end         
         
         else
         end
         
        %actionref=actioncur;
        %Pacbref=Pacbcur;
        %Npsref=Npscur;
        %NpsMref=NpsMcur;
        %NpsCVref=NpsCVcur;
        %DeltaNpsref=DeltaNpscur;
        
        %=================================================================
        
        
        
    end
    totalSuccessfulUEs = totalSuccessfulUEs + successfulUEs(simulation);
    totalSuccessfulUEsM = totalSuccessfulUEsM + successfulUEsM(simulation);
    totalSuccessfulUEsH = totalSuccessfulUEsH + successfulUEsH(simulation);
    totalFailedUEsH = totalFailedUEsH + failedUEsH(simulation);
end

counterexpout=counterexp;
memoryexpout=memoryexp;
Pacbfinal=Pacbcur;
Npfinal=Npscur;

averagePerRAO = mean(statsPerRAO,3);% Matrix with the mean of:
% [1 Arrivals, 2 Total Access attempts, 3 Collisions, 4 Successfully
% decoded preambles, 5 successful accesses per RAO]

avPreamStatsPerRAO = mean(preambleStatsPerRAO,3); %Matrix with the:
% [1 Successful reambles 2 Not used preambles 3 Collided preambles]

%% ---------- Recalculate TotalUEs and totalUEsH2H
totalUEsH = totalSuccessfulUEsH + totalFailedUEsH; % this because we stop the simulation when the M2M traffic finishes
totalUEs = totalUEsM*numSimulations + totalUEsH; %total number of UEs in the entire simulation

Ps = totalSuccessfulUEs/(totalUEs);  % Calculate the access success probability
disp('Access success probability:')
disp(Ps)

PsM = totalSuccessfulUEsM/(totalUEsM*numSimulations);  % Calculate the access success probability
disp('Access success probability M2M:')
disp(PsM)

PsH = totalSuccessfulUEsH/(totalUEsH);  % Calculate the access success probability
disp('Access success probability H2H:')
disp(PsH)

% =========================================================================

K = K./(sum(successfulUEs));  % Calculate the pmf of preamble transmissions for the successful accesses
EK = (1:RACHConfig.maxNumPreambleTxAttempts)*K;   % Calculate the average number of preamble transmissions for the successful accesses
disp('Average number of preamble transmissions:')
disp(EK)
xiK = 1:0.01:RACHConfig.maxNumPreambleTxAttempts;
Kq = interp1(1:RACHConfig.maxNumPreambleTxAttempts,cumsum(K),xiK);
% ---------------------- Percentiles ------------------
indK95 = find(Kq<=0.95, 1, 'last' ); if(isempty(indK95)), K95 = 0; else, K95 = xiK(indK95); end
indK50 = find(Kq<=0.50, 1, 'last' ); if(isempty(indK50)), K50 = 0; else, K50 = xiK(indK50); end
indK10 = find(Kq<=0.10, 1, 'last' ); if(isempty(indK10)), K10 = 0; else, K10 = xiK(indK10); end

KM = KM./(sum(successfulUEsM));  % Calculate the pmf of preamble transmissions for the successful accesses
EKM = (1:RACHConfig.maxNumPreambleTxAttempts)*KM;   % Calculate the average number of preamble transmissions for the successful accesses
disp('Average number of preamble transmissions M2M:')
disp(EKM)
xiKM = 1:0.01:RACHConfig.maxNumPreambleTxAttempts;
KqM = interp1(1:RACHConfig.maxNumPreambleTxAttempts,cumsum(KM),xiKM);
% ---------------------- Percentiles ------------------
indK95M = find(KqM<=0.95, 1, 'last' ); if(isempty(indK95M)), K95M = 0; else, K95M = xiKM(indK95M); end
indK50M = find(KqM<=0.50, 1, 'last' ); if(isempty(indK50M)), K50M = 0; else, K50M = xiKM(indK50M); end
indK10M = find(KqM<=0.10, 1, 'last' ); if(isempty(indK10M)), K10M = 0; else, K10M = xiKM(indK10M); end

KH = KH./(sum(successfulUEsH));  % Calculate the pmf of preamble transmissions for the successful accesses
EKH = (1:RACHConfig.maxNumPreambleTxAttempts)*KH;   % Calculate the average number of preamble transmissions for the successful accesses
disp('Average number of preamble transmissions H2H:')
disp(EKH)
xiKH = 1:0.01:RACHConfig.maxNumPreambleTxAttempts;
KqH = interp1(1:RACHConfig.maxNumPreambleTxAttempts,cumsum(KH),xiKH);
% ---------------------- Percentiles ------------------
indK95H = find(KqH<=0.95, 1, 'last' ); if(isempty(indK95H)), K95H = 0; else, K95H = xiKH(indK95H); end
indK50H = find(KqH<=0.50, 1, 'last' ); if(isempty(indK50H)), K50H = 0; else, K50H = xiKH(indK50H); end
indK10H = find(KqH<=0.10, 1, 'last' ); if(isempty(indK10H)), K10H = 0; else, K10H = xiKH(indK10H); end
% =========================================================================

D = D./(sum(successfulUEs));  % Calculate the pmf of access delay
Dmax = find(D>0,1,'last');
D(Dmax+1:end) = [];
ED = (1:length(D))*D;   % Calculate the average access delay
disp('Average access delay [ms]:')
disp(ED)
xiD = 1:0.1:Dmax;
Dq = interp1(1:Dmax,cumsum(D),xiD);
% ---------------------- Percentiles ------------------
indD95 = find(Dq<=0.95, 1, 'last' ); if(isempty(indD95)), D95 = 0; else, D95 = xiD(indD95); end
indD50 = find(Dq<=0.50, 1, 'last' ); if(isempty(indD50)), D50 = 0; else, D50 = xiD(indD50); end
indD10 = find(Dq<=0.10, 1, 'last' ); if(isempty(indD10)), D10 = 0; else, D10 = xiD(indD10); end

DM = DM./(sum(successfulUEsM));  % Calculate the pmf of access delay
DmaxM = find(DM>0,1,'last');
DM(DmaxM+1:end) = [];
EDM = (1:length(DM))*DM;   % Calculate the average access delay
disp('Average access delay M2M [ms]:')
disp(EDM)
xiDM = 1:0.1:DmaxM;
DqM = interp1(1:DmaxM,cumsum(DM),xiDM);
% ---------------------- Percentiles ------------------
indD95M = find(DqM<=0.95, 1, 'last' ); if(isempty(indD95M)), D95M = 0; else, D95M = xiDM(indD95M); end
indD50M = find(DqM<=0.50, 1, 'last' ); if(isempty(indD50M)), D50M = 0; else, D50M = xiDM(indD50M); end
indD10M = find(DqM<=0.10, 1, 'last' ); if(isempty(indD10M)), D10M = 0; else, D10M = xiDM(indD10M); end

DH = DH./(sum(successfulUEsH));  % Calculate the pmf of access delay
DmaxH = find(DH>0,1,'last');
DH(DmaxH+1:end) = [];
EDH = (1:length(DH))*DH;   % Calculate the average access delay
disp('Average access delay H2H [ms]:')
disp(EDH)
xiDH = 1:0.1:DmaxH;
DqH = interp1(1:DmaxH,cumsum(DH),xiDH);
% ---------------------- Percentiles ------------------
indD95H = find(DqH<=0.95, 1, 'last' ); if(isempty(indD95H)), D95H = 0; else, D95H = xiDH(indD95H); end
indD50H = find(DqH<=0.50, 1, 'last' ); if(isempty(indD50H)), D50H = 0; else, D50H = xiDH(indD50H); end
indD10H = find(DqH<=0.10, 1, 'last' ); if(isempty(indD10H)), D10H = 0; else, D10H = xiDH(indD10H); end

% filename = strcat('nov16_5000_',num2str(period),'_ltea_acb_05_4.mat');
% % 
% save(filename)
end
