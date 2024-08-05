import torch
import numpy as np
import copy
from datasets import Dataset
import numpy as np
import torch
import torch.nn as nn
from trl.core import LengthSampler

class WARP:
    def __init__(self, model, tokenizer, reward_pipeline,
                 prompt_dataset, optimizer,
                 I=2, M=2, T=100, mu=0.01, eta=0.5, beta=0.3,
                 output_min_length=10, output_max_length=20,
                 generation_min_length=40, generation_max_length=45):

        self.sft_weights_path = "sft_weights.pt"

        self.model = model
        torch.save(self.model.state_dict(), self.sft_weights_path)
        self.theta_sft = torch.load(self.sft_weights_path)
        self.tokenizer = tokenizer
        self.reward_pipeline = reward_pipeline
        self.prompt_dataset = prompt_dataset
        self.optimizer = optimizer
        self.I = I
        self.M = M
        self.T = T
        self.mu = mu
        self.eta = eta
        self.beta = beta

        self.eps = 1e-18

        # Generation setup for all models
        output_length_sampler = LengthSampler(output_min_length, output_max_length)
        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": output_length_sampler()
        }
        self.scores_kwargs = {
            "min_length": generation_min_length,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": generation_max_length
        }

    def run_method(self):
        theta_init = torch.load(self.sft_weights_path)
        theta_i_slerp = copy.deepcopy(theta_init)

        self.rewards = []
        for i in range(self.I):
            print("Started i =", i)
            theta_list = []
            processes = []
            for m in range(self.M):
                print("Started m =", m)

                theta_m = copy.deepcopy(theta_init)
                theta_m_ema = copy.deepcopy(theta_init)

                theta_m_model = copy.deepcopy(self.model)
                theta_m_model.load_state_dict(theta_m)

                theta_m_ema_model = copy.deepcopy(self.model)
                theta_m_ema_model.load_state_dict(theta_m_ema)

                for t in range(self.T):
                    print("Started t =", t)
                    # Generate completion
                    x = np.random.choice(self.prompt_dataset)
                    print("Query:", x["query"])
                    y = self.generate_completion(theta_m_model, x["input_ids"])

                    # Compute reward with KL regularization
                    r_beta = self.compute_reward(y, x["input_ids"],
                                                 theta_m_model, theta_m_ema_model)
                    self.rewards.append(r_beta)

                    # Update theta_m using policy gradient
                    self.optimizer.zero_grad()
                    log_likelihood_y = self.completion_log_prob(theta_m_model, x["input_ids"], y)
                    loss = -r_beta * log_likelihood_y
                    loss.requires_grad = True
                    print("Loss: ", loss)
                    loss.backward()
                    self.optimizer.step()

                    # Update theta_m_ema using EMA
                    theta_m_ema = self.ema_update(theta_m_ema, theta_m)
                    print("Finished t =", t)

                theta_list.append(theta_m)
                print("Finished m =", m)

            # SLERP to merge M weights
            theta_i_slerp = self.slerp(theta_init, theta_list)

            # Update theta_init towards theta_i_slerp
            theta_init = self.liti_update(theta_init, theta_i_slerp)
            print("Finished i =", i)
        
        # Save final model for generation
        self.final_init_model = copy.deepcopy(self.model)
        self.final_init_model.load_state_dict(theta_init)

        # Save outputs
        output_weights = self.output_update(self.theta_sft, theta_i_slerp)
        output_model = copy.deepcopy(self.model)
        output_model.load_state_dict(output_weights)

        return output_model


    def generate_completion(self, model, x):
        # Generate completion
        y = model.generate(input_ids=x[None, :],
                            num_return_sequences=1,
                            **self.generation_kwargs)
        return y[0]

    def completion_log_prob(self, model, x, y_tokenized):
        # Compute output logits
        output = model.generate(input_ids=x[None, :],
                                      output_scores=True,
                                      return_dict_in_generate=True,
                                      **self.scores_kwargs)
        # Scores to log_probs
        probs = []
        num_generated = len(output["scores"])
        for token_id in range(num_generated):
            token_prob = torch.sigmoid(output["scores"][token_id][0])
            token_prob = np.clip(token_prob, self.eps, 1 - self.eps)
            probs.append(torch.log(token_prob))
        # Compute completion log_prob
        prompt_len = len(x)
        sum_log_prob = 0
        for token_id in range(len(y_tokenized)):
            completion_token = y_tokenized[token_id]
            sum_log_prob += probs[token_id - prompt_len][completion_token]
        return sum_log_prob


    def compute_reward(self, y_tokenized, x, theta_m_model, theta_m_ema_model):
        # Decode completion for reward model tokenizer
        y_decoded = self.tokenizer.decode(y_tokenized)
        print("Y generated: ", y_decoded)
        # Actual reward
        r_x_y = self.reward_pipeline(y_decoded)[0]["score"]
        print("Reward: ", r_x_y)
        # Get probabilities of y from models
        log_prob_theta_m = self.completion_log_prob(theta_m_model, x, y_tokenized)
        log_prob_theta_m_ema = self.completion_log_prob(theta_m_ema_model, x, y_tokenized)
        # Compute KL-divergence
        kl_div = self.beta * (log_prob_theta_m - log_prob_theta_m_ema)
        return r_x_y - kl_div

    def ema_update(self, theta_m_ema, theta_m):
        for value in theta_m_ema:
            theta_m_ema[value] = (1 - self.mu) * theta_m_ema[value] + self.mu * theta_m[value]
        return theta_m_ema

    def slerp(self, theta_init, theta_list):
        def unit_vector(vector):
            return vector / (np.linalg.norm(vector) + self.eps)

        def angle_between(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(sum(np.multiply(v1_u, v2_u)), -1.0, 1.0))

        # Supposing M = 2 (easiest case)
        theta_i_slerp = copy.deepcopy(theta_init)
        for value in theta_i_slerp:
            task_vector1 = theta_list[0][value] - theta_init[value]
            task_vector2 = theta_list[1][value] - theta_init[value]
            omega = angle_between(task_vector1, task_vector2)
            coef = np.sin(omega / 2) / np.sin(omega)
            theta_i_slerp[value] += coef * (theta_list[0][value] + theta_list[1][value])
        return theta_i_slerp

    def liti_update(self, theta_init, theta_i_slerp):
        for value in theta_init:
            theta_init[value] = (1 - self.eta) * theta_init[value] + self.eta * theta_i_slerp[value]
        return theta_init

    def output_update(self, theta_sft, theta_i_slerp):
        for value in theta_sft:
            theta_sft[value] = (1 - self.eta) * theta_sft[value] + self.eta * theta_i_slerp[value]
        return theta_sft