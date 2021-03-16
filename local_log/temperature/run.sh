# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 100 --fix_order True  --head linear --store True --save-path linear.pkl |& tee linear.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 100 --fix_order True  --head mlp --store True --save-path mlp.pkl |& tee mlp.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 100 --fix_order True  --head None --store True --save-path None.pkl |& tee None.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update GSS --mem_size 2000 --eps_mem_batch 100 --fix_order True  --store True --save-path scr_gss.pkl |& tee scr_gss.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve ASER --update ASER --mem_size 2000 --eps_mem_batch 100 --fix_order True  --store True --save-path scr_aser.pkl |& tee scr_aser.txt
#python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update ASER --mem_size 2000 --eps_mem_batch 100 --fix_order True  --store True --save-path scr_aser_update.pkl |& tee scr_aser_update.txt
#python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve ASER --update random --mem_size 2000 --eps_mem_batch 100 --fix_order True  --store True --save-path scr_aser_retrieve.pkl |& tee scr_aser_retrieve.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 100 --fix_order True |& tee 100.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 200 --fix_order True |& tee 200.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 500 --fix_order True |& tee 500.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 1000 --fix_order True |& tee 1000.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 100 --fix_order True --temp 0.02 |& tee 02.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 100 --fix_order True --temp 0.01 |& tee 01.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 100 --fix_order True --temp 0.001 |& tee 001.txt
# python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 100 --fix_order True --temp 0.2 |& tee 20.txt
python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 100 --fix_order True --temp 0.3 |& tee 30.txt
python -u general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 2000 --eps_mem_batch 100 --fix_order True --temp 0.5 |& tee 50.txt