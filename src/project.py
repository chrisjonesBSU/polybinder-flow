"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
from flow.environments.xsede import BridgesEnvironment, CometEnvironment
import os

class MyProject(FlowProject):
    pass

class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition", default="gpu", help="Specify the partition to submit to."
        )

class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition", default="gpuq", help="Specify the partition to submit to."
        )

class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition", default="batch", help="Specify the partition to submit to."
        )

# Definition of project-related labels (classification)
@MyProject.label
def sampled(job):
    return job.doc.get("done")


@MyProject.label
def initialized(job):
    if job.sp.coarse_grain == False:
        return job.isfile("init.mol2")
    else:
        return job.isfile("atomistic_gsd.gsd")

def get_gsd_file(job):
    if job.sp.signac_project and job.sp.signac_args:
        project = signac.get_project(
            root=job.sp['signac_project'], search=True
        )
        if isinstance(job.sp.signac_args, dict):
            _job = list(project.find_jobs(filter=job.sp.signac_args))[0]
        elif isinstance(job.sp.signac_args, str): # Find job using job ID
            _job = project.open_job(id=job.sp.signac_args)
        restart_file = _job.fn('restart.gsd')
    elif job.sp.slab_file:
        restart_file = job.sp.restart_file
    return restart_file, _job.doc.steps

@directives(executable="python -u")
@directives(ngpu=1)
@MyProject.operation
@MyProject.post(sampled)
def sample(job):
    from polybinder import simulate, system
    from polybinder.utils import base_units, unit_conversions
    import numpy as np
    import logging

    with job:
        print("JOB ID NUMBER:")
        print(job.id)
        job.doc["done"] = False
        logging.info("Creating system...")
        if job.sp["system_type"] != "interface":
            system_parms = system.System(
                    density = job.sp['density'],
                    molecule = job.sp['molecule'],
                    n_compounds = job.sp['n_compounds'],
                    polymer_lengths = job.sp["polymer_lengths"],
                    para_weight = job.sp['para_weight'],
                    monomer_sequence = job.sp['monomer_sequence'],
                    sample_pdi = job.doc.sample_pdi,
                    pdi = job.sp['pdi'],
                    Mn = job.sp['Mn'],
                    Mw = job.sp['Mw'],
                    seed = job.sp['system_seed']
                )
            system = system.Initializer(
                    system = system_parms,
                    system_type = job.sp["system_type"],
                    forcefield = job.sp["forcefield"],
                    remove_hydrogens = job.sp["remove_hydrogens"],
					**job.sp["kwargs"]
            )
            if any(list(job.sp["box_constraints"].values())):
                system.target_box = system.set_target_box(
                        job.sp["box_constraints"]["x"],
                        job.sp["box_constraints"]["y"],
                        job.sp["box_constraints"]["z"]
                )
            job.doc["target_volume"] = system.target_box

            ref_values = None
            auto_scale = True

            if job.isfile("restart.gsd"):
                print("Initializing simulation from a restart.gsd file")
                restart = job.fn("restart.gsd")
                n_steps = job.sp.n_steps
                shrink_kT = None
                shrink_steps = None
                shrink_period = None
            elif any([
                    all([job.sp.signac_project, job.sp.signac_args]),
                    job.sp.restart_file
                ]
            ):
                print("Initializing simulation from a restart.gsd file")
                restart, last_n_steps = get_gsd_file(job)
                print(f"Initializing from {restart}")
                n_steps = last_n_steps + job.doc.steps
                shrink_kT = None
                shrink_steps = None
                shrink_period = None
            else:
                restart = None
                n_steps = job.sp.n_steps
                shrink_kT = job.sp['shrink_kT']
                shrink_steps = job.sp['shrink_steps']
                shrink_period = job.sp['shrink_period']

            if job.sp.coarse_grain == True:
                system.coarse_grain_system(
                        ref_distance=job.sp.ref_distance,
                        ref_mass=job.sp.ref_mass,
                        bead_mapping=job.sp.bead_mapping
                )

                ref_values = {
                    "distance": job.sp.ref_distance,
                    "energy": job.sp.ref_energy,
                    "mass": job.sp.ref_mass
                }
                auto_scale = False
                cg_potentials_dir = job.sp.cg_potentials_dir
                n_steps = job.sp.n_steps

            job.doc['num_para'] = system_parms.para
            job.doc['num_meta'] = system_parms.meta
            job.doc['num_compounds'] = system_parms.n_compounds
            job.doc['polymer_lengths'] = system_parms.polymer_lengths
            job.doc["chain_sequences"] = system_parms.molecule_sequences

        elif job.sp["system_type"] == "interface":
            slab_files = []
            ref_distances = []
            if job.doc['use_signac']is True:
                signac_args = []
                if isinstance(job.sp['signac_args'], list):
                    slab_1_arg = job.sp['signac_args'][0]
                    signac_args.append(slab_1_arg)
                    if len(job.sp['signac_args']) == 2:
                        slab_2_arg = job.sp['signac_args'][1]
                        signac_args.append(slab_2_args)
                elif not isinstance(job.sp['signac_args'], list):
                    signac_args.append(job.sp['signac_args'])

                project = signac.get_project(
                        root=job.sp['signac_project'], search=True
                    )
                for arg in signac_args:
                    if isinstance(arg, dict):
                        _job = list(project.find_jobs(filter=arg))[0]
                        slab_files.append(_job.fn('restart.gsd'))
                        ref_distances.append(_job.doc['ref_distance']/10)
                    elif isinstance(arg, str): # Find job using job ID
                        _job = project.open_job(id=arg)
                        slab_files.append(_job.fn('restart.gsd'))
                        ref_distances.append(_job.doc['ref_distance']/10)
            elif job.doc['use_signac'] is False:
                slab_files.append(job.sp.slab_file)
                ref_distances.append(job.sp['reference_distance'])

            system = system.Interface(
					slabs = slab_files,
                    ref_distance = ref_distances[0],
                    gap = job.sp['interface_gap'],
					weld_axis = job.sp["weld_axis"],
            )

            job.doc['slab_ref_distances'] = system.ref_distance
            shrink_kT = None
            shrink_steps = None
            shrink_period = None

        if job.sp.coarse_grain == False:
            system.system.save('init.mol2', overwrite=True)
            cg_potentials_dir = None

        logging.info("System generated...")
        logging.info("Starting simulation...")

        simulation = simulate.Simulation(
                system,
                r_cut=job.sp["r_cut"],
                tau_kt=job.sp['tau_kt'],
		        tau_p=job.sp['tau_p'],
                nlist=job.sp['neighbor_list'],
                dt=job.sp['dt'],
                seed=job.sp['sim_seed'],
                auto_scale=auto_scale,
                ref_values=ref_values,
                mode="gpu",
                gsd_write=max([int(job.doc['steps']/100), 1]),
                log_write=max([int(job.doc['steps']/10000), 1]),
                restart=restart,
				cg_potentials_dir=cg_potentials_dir
        )

        logging.info("Simulation object generated...")
        job.doc['ref_energy'] = simulation.ref_energy
        job.doc['ref_distance'] = simulation.ref_distance
        job.doc['ref_mass'] = simulation.ref_mass
        job.doc['real_timestep'] = unit_conversions.convert_to_real_time(
				simulation.dt,
                simulation.ref_energy,
                simulation.ref_distance,
                simulation.ref_mass
        )
        job.doc['time_unit'] = 'fs'
        job.doc['steps_per_frame'] = simulation.gsd_write
        job.doc['steps_per_log'] = simulation.log_write

        if job.sp['procedure'] == "quench":
            job.doc['T_SI'] = unit_conversions.kelvin_from_reduced(
                job.sp['kT_quench'],
                simulation.ref_energy
            )
            job.doc['T_unit'] = 'K'
            logging.info("Beginning quench simulation...")
            done = simulation.quench(
                        kT = job.sp['kT_quench'],
					    pressure = job.sp['pressure'],
                        n_steps = job.sp['n_steps'],
                        shrink_kT = shrink_kT,
                        shrink_steps = shrink_steps,
                        wall_axis = job.sp['walls'],
                        shrink_period = shrink_period
            )

        elif job.sp['procedure'] == "anneal":
            if restart is not None:
                step_sequence = [
                        i + last_n_steps for i in job.sp.anneal_sequence
                ]
            else:
                step_sequence = job.sp.anneal_sequence

            logging.info("Beginning anneal simulation...")
            if not job.sp['schedule']:
                kT_list = np.linspace(job.sp['kT_anneal'][0],
                                      job.sp['kT_anneal'][1],
                                      len(job.sp['anneal_sequence']),
                )
                kT_SI = [
                        unit_conversions.kelvin_from_reduced(
                            kT, simulation.ref_energy
                        ) for kT in kT_list
                ]
                job.doc['T_SI'] = kT_SI
                job.doc['T_unit'] = 'K'

            done = simulation.anneal(
                        kT_init = job.sp['kT_anneal'][0],
                        kT_final = job.sp['kT_anneal'][1],
					    pressure = job.sp['pressure'],
                        step_sequence = step_sequence,
                        schedule = job.sp['schedule'],
                        shrink_kT = shrink_kT,
                        shrink_steps = shrink_steps,
                        wall_axis = job.sp['walls'],
                        shrink_period = shrink_period
            )
        job.doc["done"] = done

if __name__ == "__main__":
    MyProject().main()
