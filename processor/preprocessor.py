import awkward as ak
from coffea.processor import ProcessorABC
from coffea.nanoevents.methods import candidate

from .tagger_gen_matching import match_Wplus, match_Wminus, match_Z, match_QCD

class PreProcessor(ProcessorABC):
    """
    Produces a flat training ntuple from PFNano.
    """

    def __init__(self):
        pass

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        pass

        dataset = events.metadata["dataset"]

        def build_p4(obj):
            return ak.zip(
                {
                    "pt": obj.pt,
                    "eta": obj.eta,
                    "phi": obj.phi,
                    "mass": obj.mass,
                    "charge": obj.charge,
                },
                with_name="PtEtaPhiMCandidate",
                behavior=candidate.behavior,
            )

        #### Reference processor: https://github.com/jennetd/hbb-coffea/blob/master/boostedhiggs/vhbbprocessor.py
        #### Reference: HIG-24-017-paper-v23
        #### The relative isolation variable for electrons (muons) is defined as the scalar sum pT of charged hadrons and neutral particles within
        #### a cone of radius ∆R = 0.3 (0.4) around the lepton, corrected for pileup and divided by the lepton pT

        #### electrons must have pT > 10 GeV, |η| < 2.5, pass loose identification criteria, and have relative isolation less than 0.15
        electrons = events.Electron
        ele_selections = (electrons.pt > 10) & (abs(electrons.eta) < 2.5) & (electrons.cutBased >= 2) # & (electrons.pfRelIso03_all < 0.15)
        electrons = electrons[ele_selections]

        #### Muons must have pT > 10 GeV, |η | < 2.4, pass loose identification criteria, and have relative isolation less than 0.25
        muons = events.Muon
        mu_selections = (muons.pt > 10) & (abs(muons.eta) < 2.4) & (muons.looseId) & (muons.pfRelIso03_all < 0.25)
        muons = muons[mu_selections]

        #### Hadronically decaying tau leptons must have pT > 20 GeV, |η| < 2.3, and pass the DEEP TAU algorithm identification requirements
        #### In the analysisn note, (Decay Mode != 5,6,7) and tightidDeepTau
        #### Run 2 reference:
        # (events.Tau.pt > 20)
        # (abs(events.Tau.eta) < 2.3)
        # (events.Tau.rawIso < 5)
        # (events.Tau.idDeepTau2017v2p1VSjet) This was a bitmask

        # taus = events.Tau
        # tau_selections = (taus.pt > 20) & (abs(taus.eta) < 2.3) & (taus.idDeepTau2018v2p5VSjet >= 6)
        #### byDeepTau2018v2p5VSjet ID working points (deepTau2018v2p5):
        #### 1 = VVVLoose, 2 = VVLoose, 3 = VLoose, 4 = Loose, 5 = Medium, 6 = Tight, 7 = VTight, 8 = VVTight

        #### Ignore taus at the moment, too many disagreement
        leptons = ak.concatenate([electrons, muons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, ascending=False)]
        candidatelep_p4 = build_p4(leptons)

        #### Ak8 jets
        #### be separated from any isolated leptons or photons by ∆R > 0.8
        fatjets = events.FatJet
        fj_selections = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & (fatjets.isTight)
        fatjets = fatjets[fj_selections]
        fatjets = fatjets[ak.argsort(fatjets.pt, ascending=False)]

        #### We only keep jets where ALL leptons are DeltaR > 0.8 away
        dr_table = fatjets.metric_table(candidatelep_p4)
        clean_mask = ak.all(dr_table > 0.8, axis=-1)
        candidatefj = fatjets[clean_mask]

        ###### =========== Gen-level matching ===========
        genparts = events.GenPart

        if "Wto2Q" in dataset:
            wplus_genvars, _ = match_Wplus(genparts, candidatefj)
            wminus_genvars, _ = match_Wminus(genparts, candidatefj)
        elif "Zto2Q" in dataset:
            z_genvars, _ = match_Z(genparts, candidatefj)
        elif "QCD" in dataset:
            qcd_genvars, _ = match_QCD(genparts, candidatefj)