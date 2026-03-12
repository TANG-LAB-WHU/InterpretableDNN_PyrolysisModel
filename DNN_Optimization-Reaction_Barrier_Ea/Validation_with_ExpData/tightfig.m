function hfig = tightfig(hfig)
% tightfig: Alters a figure so that it has the minimum size necessary to
% enclose all axes in the figure without excess space around them.
%
% Note that tightfig will expand the figure to accommodate axes that are
% larger than the figure was initially. If you have a large figure with
% many subplots, and some of them are outside the figure boundary, using
% tightfig will expand the figure to include those axes.
%
% hfig - handle to figure, if not supplied, the current figure will be used
% instead.

if nargin == 0
    hfig = gcf;
end

% There can be an issue with tightfig when the user has been modifying
% the contnts manually, so we start from a fresh figure and copy contents over
hfig_old = hfig;
if ~ishandle(hfig_old)
    error('tightfig: input handle is not a valid figure handle');
end

% Get the axes handles
hax = findobj(hfig_old, 'type', 'axes');

% If we have no axes, there's nothing to do
if isempty(hax)
    return
end

% Some versions of MATLAB have a bug: if the figure has a uicontrol at the bottom
% and the figure is docked, then the tightfig function will move the
% controls down by the height of the taskbar (48 pixels in my case).
% We account for this bug by not allowing the figure window to
% be smaller than the current window.
old_pos = get(hfig_old, 'Position');

% Get the current axis positions
positions = cell(numel(hax), 1);
for i = 1:numel(hax)
    positions{i} = get(hax(i), 'Position');
end

% Get the figure size
orig_units = get(hfig_old, 'Units');
set(hfig_old, 'Units', 'points');
orig_pos = get(hfig_old, 'Position');
orig_size = orig_pos(3:4);

% Calculate maximum axis position extents
ax_pos = zeros(numel(positions),4);
for i = 1:numel(positions)
    ax_pos(i,:) = positions{i};
end
max_extent = [min(ax_pos(:, 1)), min(ax_pos(:, 2)), ...
    max(ax_pos(:, 1) + ax_pos(:, 3)), max(ax_pos(:, 2) + ax_pos(:, 4))];
new_size = max_extent(3:4) - max_extent(1:2);

% Calculate the adjustment needed to remove unused space
diff_size = orig_size - new_size;
diff_pos = -0.5 * [diff_size, diff_size] - max_extent(1:2);
diff_pos = [diff_pos, diff_size];

% Adjust the positions of each axis
for i = 1:numel(hax)
    set(hax(i), 'Position', positions{i} + diff_pos);
end

% Restore units
set(hfig_old, 'Units', orig_units);

% Update figure position
new_pos = orig_pos + [0, 0, -diff_size(1), -diff_size(2)];
if any(new_pos(3:4) < 1)
    % Don't allow figure to become too small
    new_pos(3:4) = max(new_pos(3:4), [1, 1]);
end
set(hfig_old, 'Position', new_pos);
end
