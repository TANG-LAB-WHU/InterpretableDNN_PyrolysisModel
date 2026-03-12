function G = Conversion(x)
% Considered gas-solid reaction models
G(:, 1) = x.^2; % Deffusion, one-dimensional (parabolic)
G(:, 2) = x + (1 - x).*log(1 -x); % Diffusion, two-demensional (Valensi equation)
G(:, 3) = (1 - (1 - x).^(1/2)).^(1/2); % Diffusion, two-dimensional (Jander equation, n = 1/2)
G(:, 4) = (1 - (1 - x).^(1/2)).^2; % Diffusion, two-dimensional (Jander equation, n = 2)
G(:, 5) = 1 - 2*x/3 - (1 - x).^(2/3); % Diffusion, three-dimensional (Ginstling-Brounshtein equation)
G(:, 6) = (1 - (1 - x).^(1/3)).^(1/2); % Diffusion, three-dimensional (Jander equation, n = 1/2)
G(:, 7) = (1 - (1 - x).^(1/3)).^2; % Diffusion, three-dimensional (Jander equation, n = 2)
G(:, 8) = ((1 + x).^(1/3) - 1).^2; % Diffusion, three-dimensional (Anti-Jander equation)
G(:, 9) = ((1 - x).^(-1/3) - 1).^2; % Diffusion, three-dimensional (Zhuralev-Lesokin-Tempelman equation)
G(:, 10) = (-log(1 - x)).^10; % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 1/10
G(:, 11) = (-log(1 - x)).^9; % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 1/9
G(:, 12) = (-log(1 - x)).^8; % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 1/8
G(:, 13) = (-log(1 - x)).^7; % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 1/7
     
G(:, 18) = (-log(1 - x)).^2; % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 1/2
G(:, 19) = (-log(1 - x)).^(3/2); % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 2/3
G(:, 20) = (-log(1 - x)).^(4/3); % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 3/4
G(:, 21) = (-log(1 - x)).^(1/1); % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 1
G(:, 22) = (-log(1 - x)).^(1/1.5); % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 1.5
G(:, 23) = (-log(1 - x)).^(1/2); % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 2
G(:, 24) = (-log(1 - x)).^(1/2.5); % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 2.5
G(:, 25) = (-log(1 - x)).^(1/3); % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 3
G(:, 26) = (-log(1 - x)).^(1/4); % Nucleation followed nuclei growth, Avrami-Erofeev euqation n = 4
G(:, 27) = log(x./(1- x)); % Nucleation followed nuclei growth, Prout-Tomkins euqation
G(:, 28) = x.^2; % Nucleation with power law, Mapel power n = 2
G(:, 29) = x.^(3/2); % Nucleation with power law, Mapel power n = 3/2
G(:, 30) = x.^1; % Nucleation with power law, Mapel power n = 1
G(:, 31) = x.^(1/2); % Nucleation with power law, Mapel power n = 1/2
G(:, 32) = x.^(1/3); % Nucleation with power law, Mapel power n = 1/3
G(:, 33) = x.^(1/4); % Nucleation with power law, Mapel power n = 1/4
G(:, 34) = 1 - (1 - x).^(1/2); % Geometrical contraction-contracting cylinder
G(:, 35) = 1 - (1 - x).^(1/3); % Geometrical contraction-contracting sphere
G(:, 36) = -log(1 - x); % First-order-based reaction (n = 1)
G(:, 37) = ((1 - x).^(-0.1) - 1)/(0.1); % OverFirst-order-based reaction (n = 1.1)
G(:, 38) = ((1 - x).^(-0.2) - 1)/(0.2); % OverFirst-order-based reaction (n = 1.2)
G(:, 39) = ((1 - x).^(-0.3) - 1)/(0.3); % OverFirst-order-based reaction (n = 1.3)
G(:, 40) = ((1 - x).^(-0.4) - 1)/(0.4); % OverFirst-order-based reaction (n = 1.4)
G(:, 41) = ((1 - x).^(-0.5) - 1)/(0.5); % OverFirst-order-based reaction (n = 1.5)
G(:, 42) = ((1 - x).^(-0.6) - 1)/(0.6); % OverFirst-order-based reaction (n = 1.6)
G(:, 43) = ((1 - x).^(-0.7) - 1)/(0.7); % OverFirst-order-based reaction (n = 1.7)
G(:, 44) = ((1 - x).^(-0.8) - 1)/(0.8); % OverFirst-order-based reaction (n = 1.8)
G(:, 45) = ((1 - x).^(-0.1) - 1)/(0.1); % OverFirst-order-based reaction (n = 1.9)
G(:, 46) = ((1 - x).^(-1) - 1)/(1); % Second-order-based reaction (n = 2)
G(:, 47) = ((1 - x).^(-1.1) - 1)/(1.1); % OverSecond-order-based reaction (n = 2.1)
G(:, 48) = ((1 - x).^(-1.2) - 1)/(1.2); % OverSecond-order-based reaction (n = 2.2)
G(:, 49) = ((1 - x).^(-1.3) - 1)/(1.3); % OverSecond-order-based reaction (n = 2.3)
G(:, 50) = ((1 - x).^(-1.4) - 1)/(1.4); % OverSecond-order-based reaction (n = 2.4)
G(:, 51) = ((1 - x).^(-1.5) - 1)/(1.5); % OverSecond-order-based reaction (n = 2.5)
G(:, 52) = ((1 - x).^(-1.6) - 1)/(1.6); % OverSecond-order-based reaction (n = 2.6)
G(:, 53) = ((1 - x).^(-1.7) - 1)/(1.7); % OverSecond-order-based reaction (n = 2.7)
G(:, 54) = ((1 - x).^(-1.8) - 1)/(1.8); % OverSecond-order-based reaction (n = 2.8)
G(:, 55) = ((1 - x).^(-1.9) - 1)/(1.9); % OverSecond-order-based reaction (n = 2.9)
G(:, 56) = ((1 - x).^(-2) - 1)/2; % Third-order-based reaction (n = 3)
G(:, 57) = ((1 - x).^(-2.1) - 1)/2.1; % OverThird-order-based reaction (n = 3.1)
G(:, 58) = ((1 - x).^(-2.2) - 1)/2.2; % OverThird-order-based reaction (n = 3.2)
G(:, 59) = ((1 - x).^(-2.3) - 1)/2.3; % OverThird-order-based reaction (n = 3.3)
G(:, 60) = ((1 - x).^(-2.4) - 1)/2.4; % OverThird-order-based reaction (n = 3.4)
G(:, 61) = ((1 - x).^(-2.5) - 1)/2.5; % OverThird-order-based reaction (n = 3.5)
G(:, 62) = ((1 - x).^(-2.6) - 1)/2.6; % OverThird-order-based reaction (n = 3.6)
G(:, 63) = ((1 - x).^(-2.7) - 1)/2.7; % OverThird-order-based reaction (n = 3.7)
G(:, 64) = ((1 - x).^(-2.8) - 1)/2.8; % OverThird-order-based reaction (n = 3.8)
G(:, 65) = ((1 - x).^(-2.9) - 1)/2.9; % OverThird-order-based reaction (n = 3.9)
G(:, 66) = ((1 - x).^(-3) - 1)/3; % Fourth-order-based reaction (n = 4)
G(:, 67) = ((1 - x).^(-3.1) - 1)/3.1; % OverFourth-order-based reaction (n = 4.1)
G(:, 68) = ((1 - x).^(-3.2) - 1)/3.2; % OverFourth-order-based reaction (n = 4.2)
G(:, 69) = ((1 - x).^(-3.3) - 1)/3.3; % OverFourth-order-based reaction (n = 4.3)
G(:, 70) = ((1 - x).^(-3.4) - 1)/3.4; % OverFourth-order-based reaction (n = 4.4)
G(:, 71) = ((1 - x).^(-3.5) - 1)/3.5; % OverFourth-order-based reaction (n = 4.5)
G(:, 72) = ((1 - x).^(-3.6) - 1)/3.6; % OverFourth-order-based reaction (n = 4.6)
G(:, 73) = ((1 - x).^(-3.7) - 1)/3.7; % OverFourth-order-based reaction (n = 4.7)
G(:, 74) = ((1 - x).^(-3.8) - 1)/3.8; % OverFourth-order-based reaction (n = 4.8)
G(:, 75) = ((1 - x).^(-3.9) - 1)/3.9; % OverFourth-order-based reaction (n = 4.9)
G(:, 76) = ((1 - x).^(-4) - 1)/4; % Fifth-order-based reaction (n = 5)
G(:, 77) = ((1 - x).^(-4.1) - 1)/4.1; % OverFifth-order-based reaction (n = 5.1)
G(:, 78) = ((1 - x).^(-4.2) - 1)/4.2; % OverFifth-order-based reaction (n = 5.2)
G(:, 79) = ((1 - x).^(-4.3) - 1)/4.3; % OverFifth-order-based reaction (n = 5.3)
G(:, 80) = ((1 - x).^(-4.4) - 1)/4.4; % OverFifth-order-based reaction (n = 5.4)
G(:, 81) = ((1 - x).^(-4.5) - 1)/4.5; % OverFifth-order-based reaction (n = 5.5)
G(:, 82) = ((1 - x).^(-4.6) - 1)/4.6; % OverFifth-order-based reaction (n = 5.6)
G(:, 83) = ((1 - x).^(-4.7) - 1)/4.7; % OverFifth-order-based reaction (n = 5.7)
G(:, 84) = ((1 - x).^(-4.8) - 1)/4.8; % OverFifth-order-based reaction (n = 5.8)
G(:, 85) = ((1 - x).^(-4.9) - 1)/4.9; % OverFifth-order-based reaction (n = 5.9)
G(:, 86) = ((1 - x).^(-5) - 1)/5; % Sixth-order-based reaction (n = 6)
G(:, 87) = ((1 - x).^(-5.1) - 1)/5.1; % OverSixth-order-based reaction (n = 6.1)
G(:, 88) = ((1 - x).^(-5.2) - 1)/5.2; % OverSixth-order-based reaction (n = 6.2)
G(:, 89) = ((1 - x).^(-5.3) - 1)/5.3; % OverSixth-order-based reaction (n = 6.3)
G(:, 90) = ((1 - x).^(-5.4) - 1)/5.4; % OverSixth-order-based reaction (n = 6.4)
G(:, 91) = ((1 - x).^(-5.5) - 1)/5.5; % OverSixth-order-based reaction (n = 6.5)
G(:, 92) = ((1 - x).^(-5.6) - 1)/5.6; % OverSixth-order-based reaction (n = 6.6)
G(:, 93) = ((1 - x).^(-5.7) - 1)/5.7; % OverSixth-order-based reaction (n = 6.7)
G(:, 94) = ((1 - x).^(-5.8) - 1)/5.8; % OverSixth-order-based reaction (n = 6.8)
G(:, 95) = ((1 - x).^(-5.9) - 1)/5.9; % OverSixth-order-based reaction (n = 6.9)
G(:, 96) = ((1 - x).^(-6) - 1)/6; % Seventh-order-based reaction (n = 7)
G(:, 97) = ((1 - x).^(-7) - 1)/7; % Eighth-order-based reaction (n = 8)
G(:, 98) = ((1 - x).^(-8) - 1)/8; % Ninth-order-based reaction (n = 9)
G(:, 99) = ((1 - x).^(-9) - 1)/9; % Tenth-order-based reaction (n = 10)
G(:, 100) = ((1 - x).^(-10) - 1)/10; % Eleventh-order-based reaction (n = 11)
G(:, 101) = ((1 - x).^(-11) - 1)/11; % Twelfth-order-based reaction (n = 12)
G(:, 102) = ((1 - x).^(-12) - 1)/12; % Thirteenth-order-based reaction (n = 13)
G(:, 103) = ((1 - x).^(-13) - 1)/13; % Fourteenth-order-based reaction (n = 14)
G(:, 104) = ((1 - x).^(-14) - 1)/14; % Fifteenth-order-based reaction (n = 15)
end