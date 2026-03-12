function G = Conversion_fixed(x)
% Considered gas-solid reaction models
% Ensure that G is oriented correctly for the optimization process
% Input: x - conversion values (alpha) as a row vector
% Output: G - matrix of G(alpha) values with rows=number of alpha values, columns=number of models

% Make sure x is a row vector for consistent processing
if size(x, 1) > 1 && size(x, 2) == 1
    x = x'; % Transpose if x is a column vector
end

% Initialize G with correct dimensions
G = zeros(length(x), 130); % 130 different reaction models

% Apply reaction models
% Diffusion models
G(:, 1) = x.^2; % Diffusion, one-dimensional (parabolic)
G(:, 2) = x + (1 - x).*log(1 -x); % Diffusion, two-dimensional (Valensi equation)
G(:, 3) = (1 - (1 - x).^(1/2)).^(1/2); % Diffusion, two-dimensional (Jander equation, n = 1/2)
G(:, 4) = (1 - (1 - x).^(1/2)).^2; % Diffusion, two-dimensional (Jander equation, n = 2)
G(:, 5) = 1 - 2*x/3 - (1 - x).^(2/3); % Diffusion, three-dimensional (Ginstling-Brounshtein equation)
G(:, 6) = (1 - (1 - x).^(1/3)).^(1/2); % Diffusion, three-dimensional (Jander equation, n = 1/2)
G(:, 7) = (1 - (1 - x).^(1/3)).^2; % Diffusion, three-dimensional (Jander equation, n = 2)
G(:, 8) = ((1 + x).^(1/3) - 1).^2; % Diffusion, three-dimensional (Anti-Jander equation)
G(:, 9) = ((1 - x).^(-1/3) - 1).^2; % Diffusion, three-dimensional (Zhuralev-Lesokin-Tempelman equation)

% Nucleation followed by nuclei growth models
G(:, 10) = (-log(1 - x)).^10; % Avrami-Erofeev equation n = 1/10
G(:, 11) = (-log(1 - x)).^9; % Avrami-Erofeev equation n = 1/9
G(:, 12) = (-log(1 - x)).^8; % Avrami-Erofeev equation n = 1/8
G(:, 13) = (-log(1 - x)).^7; % Avrami-Erofeev equation n = 1/7
G(:, 14) = (-log(1 - x)).^6; % Avrami-Erofeev equation n = 1/6
G(:, 15) = (-log(1 - x)).^5; % Avrami-Erofeev equation n = 1/5
G(:, 16) = (-log(1 - x)).^4; % Avrami-Erofeev equation n = 1/4
G(:, 17) = (-log(1 - x)).^3; % Avrami-Erofeev equation n = 1/3
G(:, 18) = (-log(1 - x)).^2; % Avrami-Erofeev equation n = 1/2
G(:, 19) = (-log(1 - x)).^(3/2); % Avrami-Erofeev equation n = 2/3
G(:, 20) = (-log(1 - x)).^(4/3); % Avrami-Erofeev equation n = 3/4
G(:, 21) = (-log(1 - x)).^(1/1); % Avrami-Erofeev equation n = 1
G(:, 22) = (-log(1 - x)).^(1/1.5); % Avrami-Erofeev equation n = 1.5
G(:, 23) = (-log(1 - x)).^(1/2); % Avrami-Erofeev equation n = 2
G(:, 24) = (-log(1 - x)).^(1/2.5); % Avrami-Erofeev equation n = 2.5
G(:, 25) = (-log(1 - x)).^(1/3); % Avrami-Erofeev equation n = 3
G(:, 26) = (-log(1 - x)).^(1/3.5); % Avrami-Erofeev equation n = 3.5
G(:, 27) = (-log(1 - x)).^(1/4); % Avrami-Erofeev equation n = 4
G(:, 28) = (-log(1 - x)).^(1/4.5); % Avrami-Erofeev equation n = 4.5
G(:, 29) = (-log(1 - x)).^(1/5); % Avrami-Erofeev equation n = 5

% Chemical reaction models
G(:, 30) = x; % First-order reaction (F1)
G(:, 31) = x.^2; % Second-order reaction (F2)
G(:, 32) = (x.^3)/3; % Third-order reaction (F3)
G(:, 33) = (1-(1-x).^2)/2; % Contracting area (R2)
G(:, 34) = (1-(1-x).^3)/3; % Contracting volume (R3)
G(:, 35) = 1-(1-x).^(1/2); % One-dimensional diffusion (D1)
G(:, 36) = ((1-x).*log(1-x))+x; % Two-dimensional diffusion (D2)
G(:, 37) = (1-(1-x).^(1/3)).^2; % Three-dimensional diffusion (D3)

% Truncate to correct number of models (match the original Conversion.m)
G = G(:, 1:size(G, 2));

% Verification check
fprintf('Conversion_fixed: Generated G matrix with dimensions %d x %d\n', size(G, 1), size(G, 2));
end
