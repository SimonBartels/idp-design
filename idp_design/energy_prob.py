import pdb
import numpy as onp
import pandas as pd
import random
import itertools
import unittest
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax_md import space

from idp_design.utils import RES_ALPHA, NUM_RESIDUES, N20
import idp_design.utils as utils

jax.config.update("jax_enable_x64", True)


mapped_wang_frenkel = vmap(utils.wang_frenkel, in_axes=(None, 0, 0, 0, 0 ,0))
mapped_coul = vmap(utils.coul, in_axes=(None, 0, None, None, 0))


def get_energy_fn(bonded_nbrs, base_unbonded_nbrs, displacement_fn, fully_probabilistic, deterministic_i,
                  deterministic_j, fully_deterministic, use_gg=True):

    if use_gg:
        default_debye_kappa = utils.DEBYE_KAPPA_GG
        debye_path = utils.DEBYE_GG_PATH
        wf_path = utils.WF_GG_PATH
    else:
        default_debye_kappa = utils.DEBYE_KAPPA
        debye_path = utils.DEBYE_PATH
        wf_path = utils.WF_PATH

    eps_table, sigma_table, nu_table, mu_table, rc_table = utils.read_wf(fpath=wf_path)
    debye_rc_table = utils.read_debye(fpath=debye_path)

    # spring_k = 8.03 * 10 / 4.184 # FIXME: double check
    spring_k = 9.6 * 2

    # Flatten the parameter tables
    eps_flattened = list()
    sigma_flattened = list()
    nu_flattened = list()
    mu_flattened = list()
    rc_flattened = list()
    debye_rc_flattened = list()
    all_pair_charges = list()
    for i in range(NUM_RESIDUES):
        pair_charges = []
        for j in range(NUM_RESIDUES):
            eps_flattened.append(eps_table[i][j])
            sigma_flattened.append(sigma_table[i][j])
            nu_flattened.append(nu_table[i][j])
            mu_flattened.append(mu_table[i][j])
            rc_flattened.append(rc_table[i][j])

            debye_rc_flattened.append(debye_rc_table[i][j])

            pair_charges.append([utils.charges[i], utils.charges[j]])
        all_pair_charges.append(pair_charges)

    eps_flattened = jnp.array(eps_flattened, dtype=jnp.float64)
    sigma_flattened = jnp.array(sigma_flattened, dtype=jnp.float64)
    nu_flattened = jnp.array(nu_flattened, dtype=jnp.float64)
    mu_flattened = jnp.array(mu_flattened, dtype=jnp.float64)
    rc_flattened = jnp.array(rc_flattened, dtype=jnp.float64)
    debye_rc_flattened = jnp.array(debye_rc_flattened, dtype=jnp.float64)
    all_pair_charges = jnp.array(all_pair_charges)
    pair_charges = all_pair_charges.reshape([all_pair_charges.shape[0]*all_pair_charges.shape[1], 2])

    def subterms_fn(R, pseq, unbonded_nbrs=base_unbonded_nbrs, debye_kappa=default_debye_kappa):
        ub_i = unbonded_nbrs[:, 0]
        ub_j = unbonded_nbrs[:, 1]
        mask = jnp.array(ub_i < R.shape[0], dtype=jnp.int32)

        def make_pairwise_unbonded():
            def pairwise_unbonded(i, j):
                ipos = R[i]
                jpos = R[j]

                r = space.distance(displacement_fn(ipos, jpos))

                iprobs = pseq[i]
                jprobs = pseq[j]
                all_probs = jnp.kron(iprobs, jprobs)

                # if i < linker_start or i >= linker_end:
                #     aa = jnp.argmax(pseq[i])
                #     wf_vals = mapped_wang_frenkel(
                #         r, rc_table[aa], sigma_table[aa],
                #         nu_table[aa], mu_table[aa],
                #         eps_table[aa])
                #     wf_val = jnp.dot(jprobs, wf_vals)
                # elif j < linker_start or j >= linker_end:
                #     aa = jnp.argmax(pseq[j])
                #     wf_vals = mapped_wang_frenkel(
                #         r, rc_table[aa], sigma_table[aa],
                #         nu_table[aa], mu_table[aa],
                #         eps_table[aa])
                #     wf_val = jnp.dot(iprobs, wf_vals)
                # else:
                wf_vals = mapped_wang_frenkel(
                    r, rc_flattened, sigma_flattened,
                    nu_flattened, mu_flattened,
                    eps_flattened)
                wf_val = jnp.dot(all_probs, wf_vals)
                #print(wf_val)
                # TODO: It seems the line below is faster!
                #wf_val = jnp.dot(iprobs, jnp.dot(jnp.reshape(wf_vals, [iprobs.shape[0], jprobs.shape[0]]), jprobs))
                #print(wf_val)

                coul_vals = mapped_coul(r, pair_charges,
                                        utils.debye_relative_dielectric, debye_kappa,
                                        debye_rc_flattened)
                coul_val = jnp.dot(all_probs, coul_vals)
                #print(coul_val)
                # TODO: It seems the line below is faster!
                #coul_val = jnp.dot(iprobs, jnp.dot(jnp.reshape(coul_vals, [iprobs.shape[0], jprobs.shape[0]]), jprobs))
                #print(coul_val)

                val = wf_val + coul_val
                return val, (wf_val, coul_val)

            def pairwise_unbonded_deterministic_i(i, j):
                ipos = R[i]
                jpos = R[j]

                r = space.distance(displacement_fn(ipos, jpos))

                aa = jnp.argmax(pseq[i])
                jprobs = pseq[j]

                wf_vals = mapped_wang_frenkel(
                    r, rc_table[aa], sigma_table[aa],
                    nu_table[aa], mu_table[aa],
                    eps_table[aa])
                wf_val = jnp.dot(jprobs, wf_vals)
                #print(wf_val)
                # TODO: It seems the line below is faster!
                #wf_val = jnp.dot(iprobs, jnp.dot(jnp.reshape(wf_vals, [iprobs.shape[0], jprobs.shape[0]]), jprobs))
                #print(wf_val)

                coul_vals = mapped_coul(r, all_pair_charges[aa],
                                        utils.debye_relative_dielectric, debye_kappa,
                                        debye_rc_table[aa])
                coul_val = jnp.dot(jprobs, coul_vals)
                # TODO: implement for real implementation!
                coul_val = wf_val
                #print(coul_val)
                # TODO: It seems the line below is faster!
                #coul_val = jnp.dot(iprobs, jnp.dot(jnp.reshape(coul_vals, [iprobs.shape[0], jprobs.shape[0]]), jprobs))
                #print(coul_val)

                val = wf_val + coul_val
                return val, (wf_val, coul_val)

            def pairwise_unbonded_deterministic(i, j):
                ipos = R[i]
                jpos = R[j]

                r = space.distance(displacement_fn(ipos, jpos))

                aa = jnp.argmax(pseq[i])
                aa_ = jnp.argmax(pseq[j])

                wf_val = utils.wang_frenkel(
                    r, rc_table[aa, aa_], sigma_table[aa, aa_],
                    nu_table[aa, aa_], mu_table[aa, aa_],
                    eps_table[aa, aa_])

                coul_val = utils.coul(r, [utils.charges[aa], utils.charges[aa_]],
                                        utils.debye_relative_dielectric, debye_kappa,
                                        debye_rc_table[aa, aa_])

                val = wf_val + coul_val
                return val, (wf_val, coul_val)

            return pairwise_unbonded, pairwise_unbonded_deterministic_i, pairwise_unbonded_deterministic
        pairwise_unbonded, pairwise_unbonded_deterministic_i, pairwise_unbonded_deterministic = make_pairwise_unbonded()

        def pairwise_bonded(i, j):
            ipos = R[i]
            jpos = R[j]

            r = space.distance(displacement_fn(ipos, jpos))

            return utils.harmonic_spring(r, r0=utils.spring_r0, k=spring_k)

        total_bonded_val = jnp.sum(vmap(pairwise_bonded)(bonded_nbrs[:, 0], bonded_nbrs[:, 1]))
        #return total_bonded_val, (total_bonded_val, None, None, None)

        #all_unbonded_vals, (wf_val, coul_val) = vmap(pairwise_unbonded)(unbonded_nbrs[:, 0], unbonded_nbrs[:, 1])
        all_unbonded_vals, (wf_val, coul_val) = vmap(pairwise_unbonded)(fully_probabilistic[:, 0], fully_probabilistic[:, 1])
        all_unbonded_vals_i, (wf_val_, coul_val_) = vmap(pairwise_unbonded_deterministic_i)(deterministic_i[:, 0], deterministic_i[:, 1])
        all_unbonded_vals_j, (wf_val_, coul_val_) = vmap(pairwise_unbonded_deterministic_i)(deterministic_j[:, 1], deterministic_j[:, 0])
        all_unbonded_vals_, (wf_val_, coul_val_) = vmap(pairwise_unbonded_deterministic)(fully_deterministic[:, 0], fully_deterministic[:, 1])
        total_unbonded_val = jnp.sum(all_unbonded_vals) + jnp.sum(all_unbonded_vals_i) + jnp.sum(all_unbonded_vals_j) + jnp.sum(all_unbonded_vals_)
        total_wf_val = None
        total_coul_val = None
        # TODO: reapply mask!
        #total_unbonded_val = jnp.where(mask, all_unbonded_vals, 0.0).sum()
        ## total_unbonded_val = jnp.sum(all_unbonded_vals)

        ## total_wf_val = jnp.sum(wf_val)
        #total_wf_val = jnp.where(mask, wf_val, 0.0).sum()

        ## total_coul_val = jnp.sum(coul_val)
        #total_coul_val = jnp.where(mask, coul_val, 0.0).sum()

        total_energy = total_bonded_val + total_unbonded_val

        return total_energy, (total_bonded_val, total_unbonded_val, total_wf_val, total_coul_val)

    def energy_fn(R, pseq, unbonded_nbrs=base_unbonded_nbrs, debye_kappa=default_debye_kappa):
        total_energy, _ = subterms_fn(R, pseq, unbonded_nbrs, debye_kappa)
        return total_energy

    return subterms_fn, energy_fn

class TestEnergyCalculator(unittest.TestCase):

    def lammps_test(self, data_fpath, log_fpath, traj_fpath, tol_places, use_gg,
                    show_plot=False, out_box_size=500.0, r_cutoff=100.0, dr_threshold=0.2):
        displacement_fn, shift_fn = space.free()

        all_bonds, seq, atom_type_masses, num_atoms = utils.read_data_file(data_fpath)
        log_df = utils.read_log_file(log_fpath)
        traj_positions, traj_timesteps = utils.read_traj_file(traj_fpath, num_atoms)


        bonded_nbrs = jnp.array(all_bonds)
        unbonded_nbrs = list()
        for pair in itertools.combinations(jnp.arange(num_atoms), 2):
            unbonded_nbrs.append(pair)
        unbonded_nbrs = jnp.array(unbonded_nbrs)

        unbonded_nbrs_set = set([tuple(pr) for pr in onp.array(unbonded_nbrs)])
        bonded_nbrs_set = set([tuple(pr) for pr in onp.array(bonded_nbrs)])
        unbonded_nbrs = jnp.array(list(unbonded_nbrs_set - bonded_nbrs_set))
        pseq = jnp.array(utils.seq_to_one_hot(seq))

        subterms_fn, energy_fn = get_energy_fn(bonded_nbrs, unbonded_nbrs, displacement_fn, use_gg)
        subterms_fn = jit(subterms_fn)

        num_frames = len(traj_timesteps)
        all_diffs = list()

        for i in range(num_frames):
            if i == 0:
                continue

            R = traj_positions[i]
            timestep = traj_timesteps[i]

            calc_energy, (total_bonded_val, total_unbonded_val, wf_val, coul_val) = subterms_fn(R, pseq)
            print(f"\nCalc. Energy: {calc_energy}")
            print(f"\t- Bonded: {total_bonded_val}")
            print(f"\t- Unbonded: {total_unbonded_val}")
            print(f"\t\t- WF Val: {wf_val}")
            print(f"\t\t- Coulomb Val: {coul_val}")

            timestep_df = log_df[log_df.Step == timestep]
            assert(len(timestep_df) == 1)
            timestep_row = timestep_df.iloc[0]
            ref_energy = timestep_row.PotEng
            print(f"Reference Energy: {ref_energy}")

            if "E_bond" in timestep_row:
                ref_bonded_energy = timestep_row.E_bond
                print(f"\t- Bonded: {ref_bonded_energy}")
            if "E_coul" in timestep_row and "E_vdwl" in timestep_row:
                ref_coul_energy = timestep_row.E_coul
                ref_vdwl_energy = timestep_row.E_vdwl
                ref_unbonded_energy = ref_coul_energy + ref_vdwl_energy

                print(f"\t- Unbonded: {ref_unbonded_energy}")
                print(f"\t\t- WF Val: {ref_vdwl_energy}")
                print(f"\t\t- Coulomb Val: {ref_coul_energy}")


            diff = calc_energy - ref_energy
            print(f"Diff: {diff}")
            if "E_bond" in timestep_row:
                bonded_diff = total_bonded_val - ref_bonded_energy
                print(f"\t- Bonded: {bonded_diff}")
            if "E_coul" in timestep_row and "E_vdwl" in timestep_row:
                unbonded_diff = total_unbonded_val - ref_unbonded_energy
                print(f"\t- Unbonded: {unbonded_diff}")

                wf_diff = wf_val - ref_vdwl_energy
                coul_diff = coul_val - ref_coul_energy
                print(f"\t\t- WF Val: {wf_diff}")
                print(f"\t\t- Coulomb Val: {coul_diff}")

            all_diffs.append(diff)
            self.assertAlmostEqual(calc_energy, ref_energy, places=tol_places)

        if show_plot:
            plt.hist(all_diffs)
            plt.show()


    # @unittest.skip("Slow on CPU")
    def test_lammps_ref(self, tol_places=3):

        """
        data_fpath = "refdata/LAMMPS/20231101_single_chain/ACTR/ACTR.dat"
        log_fpath = "refdata/LAMMPS/20231101_single_chain/ACTR/log.lammps"
        traj_fpath = "refdata/LAMMPS/20231101_single_chain/ACTR/result.lammpstrj"
        """

        """
        data_fpath = "refdata/LAMMPS/20231101_single_chain/K25/K25.dat"
        log_fpath = "refdata/LAMMPS/20231101_single_chain/K25/log.lammps"
        traj_fpath = "refdata/LAMMPS/20231101_single_chain/K25/result.lammpstrj"
        """


        lammps_tests = list()

        # Mpipi-GG
        basedir = Path("refdata/MPIPI-GG/P53")
        data_fpath = basedir / "P53.dat"
        log_fpath = basedir / "log.lammps"
        traj_fpath = basedir / "result.lammpstrj"

        # lammps_tests.append((data_fpath, log_fpath, traj_fpath, 3, True))

        # Higher precision
        basedir = Path("refdata/LAMMPS/20231102_sc_high_precision/K25/")
        data_fpath = basedir / "K25.dat"
        log_fpath = basedir / "log.lammps"
        traj_fpath = basedir / "result.lammpstrj"
        lammps_tests.append((data_fpath, log_fpath, traj_fpath, 3, False))


        for data_fpath, log_fpath, traj_fpath, tol_places, use_gg in lammps_tests:
            self.lammps_test(data_fpath, log_fpath, traj_fpath, tol_places, use_gg, show_plot=False)




    def brute_force(self, pseq, R, bonded_nbrs, unbonded_nbrs, displacement_fn):
        n = pseq.shape[0]

        subterms_fn, energy_fn = get_energy_fn(bonded_nbrs, unbonded_nbrs, displacement_fn)
        subterms_fn = jit(subterms_fn)

        residue_combinations = list(itertools.product(*[range(NUM_RESIDUES) for i in range(n)]))
        n_res_combinations = len(residue_combinations)

        brute_sm = 0.0
        for n_rc in tqdm(range(n_res_combinations), desc="Brute force"):
            res_combo = residue_combinations[n_rc]
            seq = ""
            pr_seq = 1.0
            for i in range(n):
                residue = RES_ALPHA[res_combo[i]]
                seq += residue

                pr_seq *= pseq[i, res_combo[i]]

            pseq_oh = jnp.array(utils.seq_to_one_hot(seq))
            val, _ = subterms_fn(R, pseq_oh)
            brute_sm += val*pr_seq

        calc_energy, (total_bonded_val, total_unbonded_val, wf_val, coul_val) = subterms_fn(R, jnp.array(pseq))

        return brute_sm, calc_energy


    @unittest.skip("just for now...")
    def test_fuzzy_seq(self):
        n = 4
        R = list()
        for i in range(n):
            R.append([0.0, 0.0, utils.spring_r0*i])
        R = jnp.array(R)
        pseq = utils.random_pseq(n)
        displacement_fn, _ = space.free()

        bonded_nbrs = jnp.array([(i, i+1) for i in range(n-1)])
        unbonded_nbrs = list()
        for pair in itertools.combinations(jnp.arange(n), 2):
            unbonded_nbrs.append(pair)
        unbonded_nbrs = jnp.array(unbonded_nbrs)

        unbonded_nbrs_set = set([tuple(pr) for pr in onp.array(unbonded_nbrs)])
        bonded_nbrs_set = set([tuple(pr) for pr in onp.array(bonded_nbrs)])
        unbonded_nbrs = jnp.array(list(unbonded_nbrs_set - bonded_nbrs_set))

        brute_val, calc_val = self.brute_force(pseq, R, bonded_nbrs, unbonded_nbrs, displacement_fn)

        print(f"Calc: {calc_val}")
        print(f"Brute Force: {brute_val}")

        self.assertAlmostEqual(brute_val, calc_val)

if __name__ == "__main__":
    unittest.main()
