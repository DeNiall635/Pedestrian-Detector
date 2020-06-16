function data = NNclassifier(image, model, k)
%NN classifier: Calculates Euclidean Distance

pos_id = model.pos_images;
neg_id = model.neg_images;

% k = 1 (Nearest Neighbour)
% k = 5 (k-Nearest Neighbour)

pos_id_labels = model.pos_labels;
neg_id_labels = model.neg_labels;

image_data = [];
for j = 1 : 160
    image_row = image(j,:);
    image_data = [image_data, image_row];
end

distances = [];
pos_distances = [];
neg_distances = [];
labels = [];

%% Positive Image Classification
% ED between image and pos_images

[pos_id_rows, pos_id_columns] = size(pos_id);

final_distance_pos = 1000;


for vector_pos=1:pos_id_rows
    vector = pos_id(vector_pos,:);
	vector_label = pos_id_labels(vector_pos,:);
    % ED between each vector and image_data
    d = abs(vector-image_data);
    r=d.*d;
    s=sum(r);
    distance = sqrt(s);
    pos_distances = [pos_distances, distance];
    labels = [labels, vector_label];
end


%% Negative Image Classification

% ED between image and neg_images
[neg_id_rows, neg_id_columns] = size(neg_id);

final_distance_neg = 1000;


for vector_neg=1:neg_id_rows
    vector = neg_id(vector_neg,:);
	vector_label = neg_id_labels(vector_neg,:);
    % ED between each vector and image_data
    d = abs(vector-image_data);
    r=d.*d;
    s=sum(r);
    distance = sqrt(s);
    neg_distances = [neg_distances, distance];
    labels = [labels, vector_label];
end

%% Classification

distances = [pos_distances, neg_distances];

distance_data = [distances; labels];

% Each value in distance data has been divided by 1000
[~,inx]=sort(distance_data(1,:));
output_data = distance_data(:,inx);

% Output data in ascending order, get k values in front of the minium value
pedestrian_decider = [];

% Number of neighbours to consider:-
for i=1:k
    neighbour = output_data(:,i);
    pedestrian_decider = [pedestrian_decider, neighbour(1)];
    pedestrian = mode(pedestrian_decider);
end

%% Evaluation

if pedestrian > 0
    data = 1;
else
    data = 0;
end

end

