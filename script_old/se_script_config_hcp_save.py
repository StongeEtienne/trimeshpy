# Etienne.St-Onge@usherbrooke.ca
#  bool : False = first option (default), True = second option

bool_dict = {}
var_dict = {}

### transformation
bool_dict["lps"] = [[False, None, "--no_lps"],  # no RAS to LPS for surface
                    [True, None, "--no_xras_translation"],  # no free surfer translation error fix 
                    [True, None, "--fx"]]  # HCP flip X

var_dict["lps"] = []

### mask
bool_dict["mask"] = [[False, "--inverse_mask", "--v" ]]  # inverse index mask selection

var_dict["mask"] = [["-index", "32 67 -1"]] # label from the CC and non-"Gray matter" interface # 10 22 45

### smooth the surface
bool_dict["smooth"] = [[False, None, "--dist_weighted"],  # weight smooth by vertices distance
                       [False, None, "--area_weighted"],  # weight smooth by triangle area
                       [False, None, "--forward_step"]]  # forward step (faster but less precise) (!! no to use if step_size > 1.0 or dist or area weight is used)

var_dict["smooth"] = [["-nb_step", 2],
                      ["-step_size", 5.0]]

### surface_tracking
bool_dict["st"] = [[False, None, "--ed_not_normed"],  # end (output) direction (normals) not normed
                   [False, None, "--ed_not_weighted"]]   # end (output) direction (normals)  not weighted by area

var_dict["st"] = [["-nb_step", 100], # minimum 1
                  ["-step_size", 1.0]] # can be set to zero if no surface tracking

### dmri local tractography
bool_dict["tracto"] = [[True, "--basis dipy", "--basis mrtrix"],  # fodf input basis
                       [True, "--algo det", "--algo prob"],  # deterministic or probabilistic dMRI tractography
                       [False, "--sh_interp tl", "--sh_interp nn"],
                       [True, "--mask_interp nn", "--mask_interp tl"],
                       [False, "-inv_seed_dir", None],
                       [False, None, "-test 500"], # Test with only a subpart of the seeding points
                       [False, "-f", "-f"], # Script need it
                       [False, "--tq", "--tq"], # Script need it
                       [False, None, "--minL 0"], # test
                       [False, None, "--maxL 400"], # test
                       [False, None, "--maxL_no_dir 1"]] # test

var_dict["tracto"] = [["--step", 0.2]]

### dmri pft tractography
bool_dict["pft_tracto"] = [[True, "--basis dipy", "--basis mrtrix"],  # fodf input basis
                           [True, "--algo det", "--algo prob"],  # deterministic or probabilistic dMRI tractography
                           [False, "--sh_interp tl", "--sh_interp nn"],
                           [True, "--mask_interp nn", "--mask_interp tl"],
                           [False, "-inv_seed_dir", None],
                           [False, None, "-test 5000"], # Test with only a subpart of the seeding points
                           [False, None, "--no_pft"], # Test mask without pft
                           [False, "--all", "--all"], # Script need it
                           [False, "-f", "-f"], # Script need it
                           [False, "--tq", "--tq"], # Script need it
                           [False, None, "--minL 0"], # test
                           [False, None, "--maxL 400"], # test
                           [False, None, "--maxL_no_dir 1"], # test
                           [False, None, "--particles 15"], # test
                           [False, None, "--back 2"], # test
                           [False, None, "--front 2"], # test
                           [True, None, "--pft_theta 40"]] #test

var_dict["pft_tracto"] = [["--step", 0.2]]

### intersection
bool_dict["intersect"] = [[True, None, "-nuclei hcp/spinal_cord_lps_fx.vtk"],#pft/spinal_lps.vtk"],
                          [True, None, "-nuclei_soft hcp/gray_nuclei_lps_fx.vtk"]]#pft/gray_nuclei_lps.vtk"]]

var_dict["intersect"] = []

### tractography fusion with surface tracking
bool_dict["fusion"] = [[False, None, "-compression 0.1"]]
var_dict["fusion"] = [["-max_nb_step", 100]]

# generation of parameters string
def generate_params_dict(bool_dict, var_dict):
    params_dict = {}
    for key in bool_dict.keys():
        # initialize
        params_dict[key] = ""
        
        # add bool dict
        for bool_param in bool_dict[key]:
            if bool_param[0] is False:
                if bool_param[1] is not None:
                    params_dict[key] += bool_param[1] + " "
            else:
                if bool_param[2] is not None:
                    params_dict[key] += bool_param[2] + " "
            
        # add var dict
        for var_param in var_dict[key]:
            params_dict[key] += var_param[0] + " " + str(var_param[1]) + " "
    
    return params_dict
            
def print_param_dict(params_dict, ordered_key=None):
    print "\n### se_script_config, parameters :"
    if ordered_key is None:
        for key in params_dict.keys():
            print key
            print " ",params_dict[key]
    else:
        for key in ordered_key:
            print key
            print " ",params_dict[key]
    print "###\n"

ordered_key = ["lps", "smooth", "st", "tracto", "pft_tracto", "intersect", "fusion"]
params_dict = generate_params_dict(bool_dict, var_dict)
print_param_dict(params_dict, ordered_key)
