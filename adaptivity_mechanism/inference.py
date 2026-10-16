import torch


class AdaptiveInference:

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @staticmethod
    def confidence(logits):  #maximum softmax probability
      
        probabilities = torch.softmax(logits, dim=1)

        confidence, prediction = probabilities.max(dim=1)

        return confidence, prediction

    @torch.no_grad()
    def predict(self, x, policy):

        x = x.to(self.device,non_blocking=True)
        x = self.model.stem(x)

        # EXIT 1
        x = self.model.blocks[0](x)
        x = self.model.blocks[1](x)

        logits = self.model.exit1(x)

        confidence, prediction = self.confidence(logits)

        confidence = confidence.item()
        prediction = prediction.item()

        # Stop when
        if (
            policy.max_exit == 1                        #exit 1 is max allowed exit
            or confidence >= policy.exit1_threshold     #confidence is high enough
        ):
            return {
                "exit": 1,
                "prediction": prediction,
                "confidence": confidence,
                "battery_policy": policy.name,
            }
        
        # EXIT 2
        x = self.model.blocks[2](x)
        x = self.model.blocks[3](x)

        logits = self.model.exit2(x)

        confidence, prediction = self.confidence(logits)

        confidence = confidence.item()
        prediction = prediction.item()


        # Stop when:
        if (                                            
            policy.max_exit == 2                        #exit 2 is max allowed exit
            or confidence >= policy.exit2_threshold     #confidence is high enough
        ):
            return {
                "exit": 2,
                "prediction": prediction,
                "confidence": confidence,
                "battery_policy": policy.name,
            }

        # EXIT 3
        x = self.model.blocks[4](x)
        x = self.model.blocks[5](x)
        x = self.model.blocks[6](x)
        x = self.model.blocks[7](x)

        x = self.model.pool(x)
        x = x.flatten(1)

        logits = self.model.exit3(x)

        confidence, prediction = self.confidence(logits)

        return {
            "exit": 3,
            "prediction": prediction.item(),
            "confidence": confidence.item(),
            "battery_policy": policy.name,
        }



