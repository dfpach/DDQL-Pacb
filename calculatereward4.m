function reward = calculatereward4(NpsMcur,NpsCVcur,DeltaNpscur,Pacbcur)


if Pacbcur==1
    if NpsMcur<=3
        if NpsCVcur<0.4
            if DeltaNpscur==1
                reward=50;
            elseif DeltaNpscur==3
                reward=100;
            else
                reward=100;
            end
        else
            if DeltaNpscur==1
                reward=30;
            elseif DeltaNpscur==3
                reward=60;
            else
                reward=60;
            end            
        end
    elseif NpsMcur<7 
        if NpsCVcur<0.4
            if DeltaNpscur==1
                reward=30;
            elseif DeltaNpscur==3
                reward=60;
            else
                reward=60;
            end                        
        else
            if DeltaNpscur==1
                reward=1;
            elseif DeltaNpscur==3
                reward=15;
            else
                reward=15;
            end             
        end
    elseif NpsMcur<=10
        if NpsCVcur<0.2
            if DeltaNpscur==1
                reward=1;
            elseif DeltaNpscur==3
                reward=10;
            else
                reward=15;
            end
        else
            if DeltaNpscur==1
                reward=0.5;
            elseif DeltaNpscur==3
                reward=5;
            else
                reward=8;
            end            
        end
    else
        if NpsCVcur<0.2
            if DeltaNpscur==1
                reward=-100;
            elseif DeltaNpscur==3
                reward=-90;
            else
                reward=-85;
            end
        else
            if DeltaNpscur==1
                reward=-100;
            elseif DeltaNpscur==3
                reward=-90;
            else
                reward=-85;
            end            
        end
    end
    
elseif Pacbcur>=0.7
    if NpsMcur<=3
        if NpsCVcur<0.4
            if DeltaNpscur==1
                reward=40;
            elseif DeltaNpscur==3
                reward=80;
            else
                reward=80;
            end
        else
            if DeltaNpscur==1
                reward=25;
            elseif DeltaNpscur==3
                reward=50;
            else
                reward=50;
            end            
        end
    elseif NpsMcur<7
        if NpsCVcur<0.4
            if DeltaNpscur==1
                reward=40;
            elseif DeltaNpscur==3
                reward=70;
            else
                reward=70;
            end
        else
            if DeltaNpscur==1
                reward=5;
            elseif DeltaNpscur==3
                reward=20;
            else
                reward=20;
            end            
        end
    elseif NpsMcur<=10
        if NpsCVcur<0.2
            if DeltaNpscur==1
                reward=2;
            elseif DeltaNpscur==3
                reward=15;
            else
                reward=20;
            end
        else
            if DeltaNpscur==1
                reward=1;
            elseif DeltaNpscur==3
                reward=8;
            else
                reward=10;
            end            
        end
    else
        if NpsCVcur<0.2
            if DeltaNpscur==1
                reward=-90;
            elseif DeltaNpscur==3
                reward=-85;
            else
                reward=-80;
            end
        else
            if DeltaNpscur==1
                reward=-90;
            elseif DeltaNpscur==3
                reward=-85;
            else
                reward=-80;
            end            
        end
    end
    
elseif Pacbcur>=0.5
    if NpsMcur<=3
        if NpsCVcur<0.4
            if DeltaNpscur==1
                reward=10;
            elseif DeltaNpscur==3
                reward=5;
            else
                reward=-5;
            end             
        else
            if DeltaNpscur==1
                reward=8;
            elseif DeltaNpscur==3
                reward=3;
            else
                reward=-10;
            end               
        end
    elseif NpsMcur<7
        if NpsCVcur<0.4
            if DeltaNpscur==1
                reward=50;
            elseif DeltaNpscur==3
                reward=60;
            else
                reward=80;
            end
        else
            if DeltaNpscur==1
                reward=40;
            elseif DeltaNpscur==3
                reward=50;
            else
                reward=60;
            end            
        end
    elseif NpsMcur<=10
        if NpsCVcur<0.2
            if DeltaNpscur==1
                reward=5;
            elseif DeltaNpscur==3
                reward=20;
            else
                reward=40;
            end
        else
            if DeltaNpscur==1
                reward=5;
            elseif DeltaNpscur==3
                reward=15;
            else
                reward=30;
            end            
        end    
    else
        if NpsCVcur<0.2
            if DeltaNpscur==1
                reward=-60;
            elseif DeltaNpscur==3
                reward=-45;
            else
                reward=-40;
            end
        else
            if DeltaNpscur==1
                reward=-60;
            elseif DeltaNpscur==3
                reward=-45;
            else
                reward=-40;
            end            
        end
    end
      
elseif Pacbcur>=0.3
    if NpsMcur<=3
        if NpsCVcur<0.4
            if DeltaNpscur==1
                reward=-10;
            elseif DeltaNpscur==3
                reward=-20;
            else
                reward=-30;
            end
        else
            if DeltaNpscur==1
                reward=-15;
            elseif DeltaNpscur==3
                reward=-30;
            else
                reward=-40;
            end            
        end
    elseif NpsMcur<7
        if NpsCVcur<0.4
             if DeltaNpscur==1
                reward=50;
            elseif DeltaNpscur==3
                reward=60;
            else
                reward=80;
            end
        else
             if DeltaNpscur==1
                reward=40;
            elseif DeltaNpscur==3
                reward=50;
            else
                reward=60;
            end            
        end
    elseif NpsMcur<=10
        if NpsCVcur<0.2
             if DeltaNpscur==1
                reward=6;
            elseif DeltaNpscur==3
                reward=21;
            else
                reward=41;
            end
        else
             if DeltaNpscur==1
                reward=6;
            elseif DeltaNpscur==3
                reward=16;
            else
                reward=40;
            end            
        end
    else
        if NpsCVcur<0.2
             if DeltaNpscur==1
                reward=-50;
            elseif DeltaNpscur==3
                reward=-35;
            else
                reward=-30;
            end
        else
             if DeltaNpscur==1
                reward=-50;
            elseif DeltaNpscur==3
                reward=-35;
            else
                reward=-30;
            end            
        end
    end
    
else
    if NpsMcur<=3
        if NpsCVcur<0.4
             if DeltaNpscur==1
                reward=-20;
            elseif DeltaNpscur==3
                reward=-30;
            else
                reward=-40;
            end
        else
             if DeltaNpscur==1
                reward=-25;
            elseif DeltaNpscur==3
                reward=-40;
            else
                reward=-50;
            end            
        end
    elseif NpsMcur<7
        if NpsCVcur<0.4
             if DeltaNpscur==1
                reward=-15;
            elseif DeltaNpscur==3
                reward=-25;
            else
                reward=-35;
            end   
        else
             if DeltaNpscur==1
                reward=-20;
            elseif DeltaNpscur==3
                reward=-35;
            else
                reward=-45;
            end              
        end
    
    elseif NpsMcur<=10
        if NpsCVcur<0.2
             if DeltaNpscur==1
                reward=6;
            elseif DeltaNpscur==3
                reward=21;
            else
                reward=41;
            end 
        else
            if DeltaNpscur==1
                reward=6;
            elseif DeltaNpscur==3
                reward=16;
            else
                reward=40;
            end            
        end
    else
        if NpsCVcur<0.2
            if DeltaNpscur==1
                reward=25;
            elseif DeltaNpscur==3
                reward=15;
            else
                reward=35;
            end 
        else
            if DeltaNpscur==1
                reward=25;
            elseif DeltaNpscur==3
                reward=15;
            else
                reward=35;
            end             
        end
    end
    
end
    