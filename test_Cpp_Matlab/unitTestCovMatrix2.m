function unitTestCovMatrix2()

nbTimeStep = 1000;
nbModes = 8;
covMatrix = zeros(nbTimeStep,nbTimeStep);
dt = 0.05;

for i = 1:nbTimeStep
    for j = 1:nbTimeStep
        covMatrix(i,j) = sin(i)*cos(j)/200;
    end
end
save('covMatrix','covMatrix');

bt = zeros(nbTimeStep,nbModes);


for i = 1:nbTimeStep
    for j = 1:nbModes
        bt(i,j) = sin( i*j );
    end
end

save('bt','bt');

tau = htgenCorrelationTime2(covMatrix,bt,dt);
tau*dt

end

