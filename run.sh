GeNeVA

# port forwarding from vsky to login node hpc, then hpc to personal loaclhost
    # run the follwing from login node of hpc
ssh -N -f -L localhost:8097:localhost:8097 vsky036

    # run the follwing on your local terminal
ssh -N -f -L localhost:8097:localhost:8097 ee1180542@hpc.iitd.ac.in

# run the training command for geneva
no_proxy=localhost python geneva/inference/train.py @example_args/crim-d-subtract.args