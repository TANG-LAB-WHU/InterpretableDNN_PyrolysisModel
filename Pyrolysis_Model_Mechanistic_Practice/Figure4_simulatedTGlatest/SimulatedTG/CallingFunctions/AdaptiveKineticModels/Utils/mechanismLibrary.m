function lib = mechanismLibrary()
%MECHANISMLIBRARY  Return a cell array of all single-step kinetic mechanisms
%   Each element is a string of the form "category|modelType" so that the
%   first part matches the high-level category (diffusion, nucleation, etc.)
%   and the second part matches the specific model implemented in the
%   corresponding *Models.m file.
%
%   This helper is used by modelComparison.m to build the candidate pool
%   for GA-based multi-mechanism optimisation so that **all** mathematical
%   models defined under AdaptiveKineticModels/ are considered.
%
%   Update this list whenever you add a new model to the category files.

lib = {
  % Diffusion models
  'diffusion|parabolic_1d';
  'diffusion|valensi_2d';
  'diffusion|jander_2d';
  'diffusion|ginstling_brounshtein_3d';
  'diffusion|jander_3d';
  'diffusion|anti_jander_3d';
  'diffusion|zhuralev_lesokin_tempelman_3d';

  % Nucleation & growth models
  'nucleation|avrami_erofeev';
  'nucleation|prout_tomkins';

  % Power-law
  'powerlaw|mapel_power';

  % Geometrical contraction
  'geometrical|contracting_cylinder';
  'geometrical|contracting_sphere';

  % Reaction-order models
  'reaction_order|first_order';
  'reaction_order|nth_order';
  };
end
