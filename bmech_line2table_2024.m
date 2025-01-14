function table_data = bmech_line2table_2024(fld,ch,subjects,conditions)
% BMECH_ZOO2TABLE (fld,ch,subjects,conditions) extracts channel line data
% to Matlab table
%
% ARGUMENTS
%   fld         ...   string, folder to operate on
%   ch          ...   string, all channel name.
%   subjects    ...   string cell, subject name of all the subjects.
%   conditions  ...   string cell, all Conditions name.
%
% RETURNS
%   table_data ...  Table of channel data with subjects (second last row) and conditions (last row).
%                   If channel does not exisit it will be a vector 1xn with all values at 999.
%
%

if nargin ==2
    [subjects, conditions] = extract_filestruct(fld);
    conditions = unique(conditions);
end

fl=engine('path',fld,'ext','.zoo');
for s=1:length(subjects)
    subject = subjects{s};
    fls=fl(contains(fl,subject));
    for c=1:length(conditions)
        if s==1
            flc=fls(contains(fls,conditions{c}));
        else
            flc1=fls(contains(fls,conditions{c}));
            
            flc=[flc;flc1];
        end
        
    end
end
fl=flc;
table_data=emptytable_data(fl,ch);
row=1;
for s=1:length(subjects)
    disp(['loading all files for subject ', subjects{s}, ' ...'])
    subject = subjects{s};
    fls=fl(contains(fl,subject));
    for c=1:length(conditions)
        flc=fls(contains(fls,conditions{c}));
        condition = conditions{c};
        for i=1:length(flc)
            disp(['condition ', condition, '...'])
            data=zload(flc{i});
            disp(flc{i})
            table_data=event_extract(table_data,data,row,ch);
            table_data.Subject(row)={subject};
            table_data.Conditions(row)={condition};
            row=row+1;
        end
        disp(' ')
    end
end

function  table_event=event_extract(table_event,data,row,ch)
finame=fieldnames(data);
for ii=1:length(ch)
    event_name=[ch{ii}];
    con=contains(finame,ch{ii});
    if sum(con)>0
        table_event.(event_name)(row)={data.(ch{ii}).line};
    else
        disp(['channel does not exisit for ',ch{ii}]);
        table_event.(event_name)(row)=999*ones([1,length(data.zoosystem.Video.Indx)]);
    end
end
