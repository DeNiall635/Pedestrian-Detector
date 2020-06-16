function model = KNNmodel(pos, neg)

model.pos_images=pos;
model.neg_images=neg;

% Create an data structure below for each image
p_rows = size(pos,1);
n_rows = size(neg,1);

% Positive Images are 1 = True
model.pos_labels=ones(p_rows,1);

% Negative Images are 0 = False
model.neg_labels=zeros(n_rows,1);

end