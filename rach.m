classdef rach
    % version Nov13-2017
    % RACH channel configuration
    %   Detailed explanation goes here
    
    properties
        availablePreambles
        raoPeriodicity
        uplinkGrantsPerSubframe
        maxNumPreambleTxAttempts
        backoffIndicator
        %-----	Delay parameters	-----%
        contentionResolutionTimer
        preambleProcessingDelay
        rarProcessingDelay
        connectionRequestProcessingDelay
        maxNumMsg3Msg4TxAttempts
        harqReTxProbMsg3Msg4
        rttMsg3
        rttMsg4
    end
    
    methods
        function obj = rach(a,b,c,d,e,f,g,h,i,j,k,l,m)
            if nargin > 0
                obj.availablePreambles = a;
                obj.raoPeriodicity = b;
                obj.uplinkGrantsPerSubframe = c;
                obj.maxNumPreambleTxAttempts = d;
                obj.backoffIndicator = e;
                obj.contentionResolutionTimer = f;
                obj.preambleProcessingDelay = g;
                obj.rarProcessingDelay = h;
                obj.connectionRequestProcessingDelay = i;
                obj.maxNumMsg3Msg4TxAttempts = j;
                obj.harqReTxProbMsg3Msg4 = k;
                obj.rttMsg3 = l;
                obj.rttMsg4 = m;
            end
        end
    end
    
end